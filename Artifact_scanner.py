import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def _rgb_to_hsv_np(rgb_u8: np.ndarray):
    x = rgb_u8.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    s = np.where(mx > 1e-6, diff / (mx + 1e-6), 0.0)
    v = mx
    return s, v


def artifact_features_from_image(rgb_u8: np.ndarray) -> dict:
    x = rgb_u8.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]

    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # 0..255

    pct_white = float((lum >= 250).mean())
    pct_black = float((lum <= 5).mean())
    mean_lum = float(lum.mean())
    std_lum = float(lum.std())

    s, v = _rgb_to_hsv_np(rgb_u8)
    glare_pct = float(((v >= 0.95) & (s <= 0.20)).mean())

    exg = float((2 * g - r - b).mean())   # greener => higher
    brown = float((r - g).mean())         # browner => higher
    sky_pct = float(((b > r + 15) & (b > g + 15) & (lum > 160)).mean())

    H, W = lum.shape
    edge = max(2, int(round(min(H, W) * 0.03)))
    top = lum[:edge, :]
    bottom = lum[-edge:, :]
    left = lum[:, :edge]
    right = lum[:, -edge:]

    edge_std = float(np.mean([top.std(), bottom.std(), left.std(), right.std()]))
    edge_black = float(np.mean([(top <= 5).mean(), (bottom <= 5).mean(),
                                (left <= 5).mean(), (right <= 5).mean()]))
    edge_white = float(np.mean([(top >= 250).mean(), (bottom >= 250).mean(),
                                (left >= 250).mean(), (right >= 250).mean()]))

    # Blur metric
    if HAVE_CV2:
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        gy = np.abs(lum[1:, :] - lum[:-1, :]).mean()
        gx = np.abs(lum[:, 1:] - lum[:, :-1]).mean()
        blur = float(gx + gy)

    return dict(
        mean_lum=mean_lum, std_lum=std_lum,
        pct_white=pct_white, pct_black=pct_black,
        glare_pct=glare_pct,
        exg=exg, brown=brown, sky_pct=sky_pct,
        edge_std=edge_std, edge_black=edge_black, edge_white=edge_white,
        blur=blur,
        H=int(H), W=int(W),
    )


def resolve_image_path(dataset_root: str, image_path: str) -> str:
    """
    Handles both:
      - image_path like 'train/xxx.jpg'  (common)
      - image_path like 'xxx.jpg'        (less common)
    """
    # Try dataset_root + image_path first
    cand1 = os.path.join(dataset_root, image_path)
    if os.path.exists(cand1):
        return cand1

    # If image_path has no 'train/' prefix, try dataset_root/train/image_path
    cand2 = os.path.join(dataset_root, "train", image_path)
    if os.path.exists(cand2):
        return cand2

    # Last resort: just return cand1 and let caller error
    return cand1


def scan_images(dataset_root: str, train_csv: str, out_csv: str, max_side: int = 768, limit: int | None = None):
    df = pd.read_csv(train_csv)
    paths = df["image_path"].drop_duplicates().tolist()
    if limit is not None:
        paths = paths[:limit]

    rows = []
    for i, p in enumerate(paths, 1):
        fp = resolve_image_path(dataset_root, p)
        try:
            with Image.open(fp) as im:
                im = im.convert("RGB")

                # Downsample for speed (metrics are stable)
                if max_side and max(im.size) > max_side:
                    scale = max_side / float(max(im.size))
                    new_w = max(1, int(round(im.size[0] * scale)))
                    new_h = max(1, int(round(im.size[1] * scale)))
                    im = im.resize((new_w, new_h), resample=Image.BILINEAR)

                rgb = np.asarray(im, dtype=np.uint8)

            feats = artifact_features_from_image(rgb)
            feats["image_path"] = p
            feats["abs_path"] = fp
            rows.append(feats)
        except Exception as e:
            rows.append({"image_path": p, "abs_path": fp, "error": str(e)})

        if i % 200 == 0:
            print(f"scanned {i}/{len(paths)}")

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, "| rows:", len(out))
    return out


def print_worst(out: pd.DataFrame, k: int = 20):
    # "worst" definitions
    candidates = {
        "glare": ("glare_pct", False),
        "clipped_white": ("pct_white", False),
        "clipped_black": ("pct_black", False),
        "sky_like": ("sky_pct", False),
        "border_black": ("edge_black", False),
        "border_white": ("edge_white", False),
        "border_uniform": ("edge_std", True),  # low std => suspicious uniform bars
        "blurry": ("blur", True),              # low blur => blurrier
    }

    for name, (col, ascending) in candidates.items():
        if col not in out.columns:
            continue
        sub = out.dropna(subset=[col]).sort_values(col, ascending=ascending).head(k)
        print(f"\n=== WORST: {name} ({col}, {'low' if ascending else 'high'} is bad) ===")
        print(sub[["image_path", col, "abs_path"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="Folder containing train.csv + train/ images")
    ap.add_argument("--out_csv", default="artifact_scan.csv")
    ap.add_argument("--max_side", type=int, default=768, help="Downsample max side for speed (0 disables)")
    ap.add_argument("--limit", type=int, default=0, help="Limit images (0 = all)")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    train_csv = os.path.join(dataset_root, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv at: {train_csv}")

    limit = None if args.limit == 0 else int(args.limit)

    out = scan_images(
        dataset_root=dataset_root,
        train_csv=train_csv,
        out_csv=args.out_csv,
        max_side=args.max_side,
        limit=limit,
    )
    print_worst(out, k=20)


if __name__ == "__main__":
    main()
