# model_zoo_v1_pycharm.py
# 4 backbones -> multi-crop embeddings -> xgb per target -> oof stacking (nnls) -> submission.csv
#
# pycharm-friendly:
# - uses argparse for local paths (data_dir, weights_dir, artifacts_dir, submission_path)
# - no /kaggle/* hardcoding
# - optional offline env toggles
# - all variable names + function names are lowercase (including module-level “constants”)

import argparse
import gc
import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import timm
import timm.data
import xgboost as xgb
from PIL import Image
from scipy.optimize import nnls
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold


# =========================
# config (all lowercase)
# =========================
seed = 123
n_folds = 5

targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
n_targets = len(targets)

idx_green = targets.index("Dry_Green_g")
idx_dead = targets.index("Dry_Dead_g")
idx_clover = targets.index("Dry_Clover_g")
idx_gdm = targets.index("GDM_g")
idx_total = targets.index("Dry_Total_g")

panorama_n_crops = 5
use_pls_on_emb = True
pls_ncomp = 16

cache_basename = "emb_all"


# =========================
# model zoo
# =========================
@dataclass(frozen=True)
class MODELCFG:
    key: str
    timm_name: str
    weights_path: str
    batch: int = 16
    n_crops: int = panorama_n_crops


def build_default_model_zoo(weights_dir: str) -> List[MODELCFG]:
    # weights_path may be absolute or relative to weights_dir
    def wp(p: str) -> str:
        pth = Path(p)
        if pth.is_absolute():
            return str(pth)
        return str(Path(weights_dir) / p)

    return [
        MODELCFG(
            key="dinov3b",
            timm_name="vit_base_patch16_dinov3.lvd1689m",
            weights_path=wp("model.safetensors"),
            batch=16,
            n_crops=panorama_n_crops,
        ),
        MODELCFG(
            key="cnxv2t",
            timm_name="convnextv2_base.fcmae",
            weights_path=wp("convnextv2_base.model.safetensors"),
            batch=16,
            n_crops=panorama_n_crops,
        ),
        MODELCFG(
            key="swincrs",
            timm_name="swinv2_base_window16_256.ms_in1k",
            weights_path=wp("swinv2_base_window16_256.model.safetensors"),
            batch=16,
            n_crops=panorama_n_crops,
        ),
        MODELCFG(
            key="eva02b",
            timm_name="eva02_base_patch16_clip_224.merged2b",
            weights_path=wp("eva02_base_patch16_clip_224.merged2b.model.safetensors"),
            batch=16,
            n_crops=panorama_n_crops,
        ),
    ]


# =========================
# env / repro
# =========================
def configure_environment(offline: bool, omp_threads: int, torch_threads: int) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)

    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(max(1, torch_threads))


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# =========================
# metrics
# =========================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    out = {t: rmse(y_true[:, j], y_pred[:, j]) for j, t in enumerate(targets)}
    out["mean"] = float(np.mean(list(out.values())))
    out["all_concat"] = rmse(y_true.reshape(-1), y_pred.reshape(-1))
    return pd.Series(out)


# =========================
# data i/o
# =========================
def read_train_image_level(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    wide = (
        df[["image_path", "target_name", "target"]]
        .drop_duplicates()
        .pivot(index="image_path", columns="target_name", values="target")
        .reset_index()
    )
    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan
    return wide


def read_test_images(test_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    test_long = pd.read_csv(test_csv)
    test_images = test_long["image_path"].drop_duplicates().tolist()
    return test_long, test_images


def build_submission(test_long: pd.DataFrame, pred_wide5: pd.DataFrame, out_path: str) -> None:
    pred_long = pred_wide5.melt(
        id_vars=["image_path"],
        value_vars=targets,
        var_name="target_name",
        value_name="target",
    )
    sub = (
        test_long.merge(pred_long, on=["image_path", "target_name"], how="left")[["sample_id", "target"]]
    )
    sub.to_csv(out_path, index=False)


def find_data_dir(start_dir: str) -> Optional[str]:
    start = Path(start_dir)
    if not start.exists():
        return None
    if start.is_file():
        start = start.parent

    # search a few levels deep for train.csv + test.csv
    for root, _, files in os.walk(str(start)):
        if "train.csv" in files and "test.csv" in files:
            return root
    return None


# =========================
# weights loader (local)
# =========================
def load_local_weights_into_timm(model, weights_path: str) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"missing weights file: {weights_path}")

    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file  # pip install safetensors

        state = load_file(weights_path)
    else:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        kk = k
        for pref in ("model.", "module.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref) :]
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[weights] loaded {weights_path} | missing={len(missing)} unexpected={len(unexpected)}")


def resolve_weights_path(path_or_dir: str, prefer_contains: Optional[str] = None) -> str:
    p = Path(path_or_dir)

    if p.is_file():
        return str(p)

    if p.is_dir():
        cands = sorted(glob.glob(str(p / "**" / "*.safetensors"), recursive=True))
        if not cands:
            raise FileNotFoundError(f"no .safetensors found under: {p}")
        if prefer_contains:
            for c in cands:
                if prefer_contains in Path(c).name:
                    return c
        return cands[0]

    raise FileNotFoundError(f"missing weights path/dir: {path_or_dir}")


def weights_fingerprint(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


# =========================
# caching
# =========================
def hash_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def emb_manifest(paths: List[str], cfg: MODELCFG) -> dict:
    return {
        "timm_name": cfg.timm_name,
        "key": cfg.key,
        "n_crops": int(cfg.n_crops),
        "paths_hash": hash_paths(paths),
        "weights_fp": weights_fingerprint(cfg.weights_path),
        "num_paths": len(paths),
        "preview": paths[:5],
    }


def load_emb_cache(cache_base: str, paths: List[str], cfg: MODELCFG) -> Optional[np.ndarray]:
    npy = cache_base + ".npy"
    man = cache_base + ".manifest.json"
    if not (os.path.exists(npy) and os.path.exists(man)):
        return None

    with open(man, "r") as f:
        m = json.load(f)

    want = emb_manifest(paths, cfg)
    for k in ("timm_name", "key", "n_crops", "paths_hash", "weights_fp", "num_paths"):
        if str(m.get(k)) != str(want.get(k)):
            return None

    arr = np.load(npy)
    if arr.shape[0] != len(paths):
        return None
    return arr.astype(np.float32)


def save_emb_cache(cache_base: str, arr: np.ndarray, paths: List[str], cfg: MODELCFG) -> None:
    np.save(cache_base + ".npy", arr.astype(np.float32))
    with open(cache_base + ".manifest.json", "w") as f:
        json.dump(emb_manifest(paths, cfg), f, indent=2)


# =========================
# panorama crops
# =========================
def panorama_square_crops(img: Image.Image, n_crops: int) -> List[Image.Image]:
    w, h = img.size
    s = min(w, h)
    if w <= s:
        return [img]

    max_left = w - s
    if n_crops <= 1:
        lefts = [max_left // 2]
    else:
        lefts = [int(round(i * max_left / (n_crops - 1))) for i in range(n_crops)]

    return [img.crop((l, 0, l + s, s)) for l in lefts]


# =========================
# embeddings (multi-crop mean pool)
# =========================
@torch.inference_mode()
def embed_timm_multicrop_mean(
    image_paths: List[str],
    img_root: str,
    cfg: MODELCFG,
    device: Optional[str] = None,
) -> np.ndarray:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(cfg.timm_name, pretrained=False, num_classes=0)
    load_local_weights_into_timm(model, cfg.weights_path)
    model.eval().to(device)

    data_config = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**data_config, is_training=False)
    use_amp = device.startswith("cuda")

    feats: List[np.ndarray] = []
    bs = int(cfg.batch)

    for i in range(0, len(image_paths), bs):
        batch_paths = image_paths[i : i + bs]

        packed: List[torch.Tensor] = []
        for p in batch_paths:
            ap = os.path.join(img_root, p)
            with Image.open(ap) as im:
                img = im.convert("RGB")
            crops = panorama_square_crops(img, n_crops=cfg.n_crops)
            packed.append(torch.stack([tfm(c) for c in crops], dim=0))  # [v,c,h,w]

        x = torch.stack(packed, dim=0)  # [b,v,c,h,w] on cpu
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w).to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                f = model(x)
        else:
            f = model(x)

        f = f.view(b, v, -1).mean(dim=1)  # [b,d]
        feats.append(f.float().cpu().numpy())

        if use_amp:
            del x, f
            torch.cuda.empty_cache()

    return np.vstack(feats).astype(np.float32)


def get_all_embeddings(all_paths: List[str], img_root: str, cache_dir: str, cfg: MODELCFG) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_base = os.path.join(cache_dir, f"{cache_basename}_{cfg.key}_c{cfg.n_crops}")

    cached = load_emb_cache(cache_base, all_paths, cfg)
    if cached is not None:
        print(f"[cache] hit: {cfg.key}")
        return cached

    print(f"[cache] miss: {cfg.key} -> computing embeddings...")
    arr = embed_timm_multicrop_mean(all_paths, img_root=img_root, cfg=cfg)
    save_emb_cache(cache_base, arr, all_paths, cfg)
    return arr


# =========================
# xgb
# =========================
def fit_xgb_one(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    xgb_params: dict,
    seed_value: int,
) -> xgb.Booster:
    dtr = xgb.DMatrix(x_tr, label=y_tr)
    dva = xgb.DMatrix(x_va, label=y_va)

    params = dict(xgb_params)
    params["seed"] = int(seed_value)

    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=5000,
        evals=[(dva, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return booster


def predict_xgb(models: List[xgb.Booster], x: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(x)
    cols: List[np.ndarray] = []
    for m in models:
        if hasattr(m, "best_iteration") and m.best_iteration is not None:
            cols.append(m.predict(d, iteration_range=(0, m.best_iteration + 1)))
        else:
            cols.append(m.predict(d))
    return np.column_stack(cols).astype(np.float32)


# =========================
# pls (optional)
# =========================
@dataclass
class plsfold:
    pls: PLSRegression


def fit_pls_on_embeddings(x: np.ndarray, y: np.ndarray, ncomp: int) -> plsfold:
    ncomp2 = int(min(ncomp, x.shape[1], max(1, x.shape[0] - 1)))
    pls = PLSRegression(n_components=ncomp2, scale=True)
    pls.fit(x, y)
    return plsfold(pls=pls)


def pls_transform(pls_fold: plsfold, x: np.ndarray) -> np.ndarray:
    return pls_fold.pls.transform(x).astype(np.float32)


# =========================
# constraints
# =========================
def enforce_sum_constraints(pred5: np.ndarray) -> np.ndarray:
    pred5 = np.asarray(pred5, dtype=np.float32).copy()
    pred5 = np.maximum(pred5, 0.0)

    g = pred5[:, idx_green]
    d = pred5[:, idx_dead]
    c = pred5[:, idx_clover]

    pred5[:, idx_gdm] = g + c
    pred5[:, idx_total] = g + d + c
    return pred5


# =========================
# stacking (nnls)
# =========================
def fit_stack_weights_nnls(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    m = np.isfinite(y) & np.all(np.isfinite(p), axis=1)
    p = p[m]
    y = y[m]

    w, _ = nnls(p, y)
    if w.sum() <= 0:
        w = np.ones(p.shape[1], dtype=np.float64) / p.shape[1]
    else:
        w = w / w.sum()
    return w.astype(np.float32)


# =========================
# main
# =========================
def main(
    data_dir: Optional[str],
    weights_dir: str,
    artifacts_dir: str,
    submission_path: str,
    enforce_constraints: bool,
    offline: bool,
    nthread: int,
) -> None:
    configure_environment(offline=offline, omp_threads=max(1, nthread), torch_threads=1)
    set_seed(seed)

    if data_dir is None:
        data_dir = find_data_dir(os.getcwd())
    if data_dir is None:
        raise FileNotFoundError(
            "could not find train.csv + test.csv. pass --data_dir pointing to the folder containing them."
        )

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    os.makedirs(artifacts_dir, exist_ok=True)

    train_wide = read_train_image_level(train_csv).sort_values("image_path").reset_index(drop=True)
    test_long, test_images = read_test_images(test_csv)
    test_images = sorted(test_images)

    train_images = train_wide["image_path"].tolist()

    y = train_wide[targets].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        bad = ~np.isfinite(y).all(axis=1)
        print(f"[warn] dropping {bad.sum()} rows with nan/inf targets")
        train_wide = train_wide.loc[~bad].reset_index(drop=True)
        train_images = train_wide["image_path"].tolist()
        y = train_wide[targets].to_numpy(dtype=np.float32)

    y_clip = np.clip(y, 0.0, None)
    y_log = np.log1p(y_clip)

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "nthread": int(max(1, nthread)),
        "seed": int(seed),
        "tree_method": "hist",
        "max_bin": 256,
    }

    model_zoo = build_default_model_zoo(weights_dir=weights_dir)

    print(f"train: {len(train_images)} | test: {len(test_images)} | models: {len(model_zoo)}")
    for cfg in model_zoo:
        print(f" - {cfg.key} {cfg.timm_name} | crops {cfg.n_crops} | batch {cfg.batch} | weights {cfg.weights_path}")

    # resolve weight paths once
    resolved_zoo: List[MODELCFG] = []
    for cfg in model_zoo:
        wpath = resolve_weights_path(cfg.weights_path, prefer_contains=cfg.key)
        resolved_zoo.append(MODELCFG(cfg.key, cfg.timm_name, wpath, batch=cfg.batch, n_crops=cfg.n_crops))
    model_zoo = resolved_zoo

    # embeddings for all images (train + test)
    all_images = train_images + test_images
    emb_tr: Dict[str, np.ndarray] = {}
    emb_te: Dict[str, np.ndarray] = {}

    for cfg in model_zoo:
        arr = get_all_embeddings(all_images, img_root=data_dir, cache_dir=artifacts_dir, cfg=cfg)
        emb_tr[cfg.key] = arr[: len(train_images)]
        emb_te[cfg.key] = arr[len(train_images) :]
        del arr
        gc.collect()

    # cv
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_pred: Dict[str, np.ndarray] = {cfg.key: np.zeros((len(train_images), n_targets), np.float32) for cfg in model_zoo}
    te_sum: Dict[str, np.ndarray] = {cfg.key: np.zeros((len(test_images), n_targets), np.float32) for cfg in model_zoo}
    fold_id = np.full(len(train_images), -1, dtype=np.int32)

    for fold_idx, (idx_tr, idx_va) in enumerate(kf.split(train_images), 1):
        print(f"\n=== fold {fold_idx}/{n_folds} ===")
        fold_id[idx_va] = fold_idx

        ytr_log = y_log[idx_tr]
        yva_log = y_log[idx_va]
        true_va = y_clip[idx_va]

        for cfg in model_zoo:
            xtr_emb = emb_tr[cfg.key][idx_tr]
            xva_emb = emb_tr[cfg.key][idx_va]
            xte_emb = emb_te[cfg.key]

            if use_pls_on_emb:
                pls = fit_pls_on_embeddings(xtr_emb, ytr_log, ncomp=pls_ncomp)
                s_tr = pls_transform(pls, xtr_emb)
                s_va = pls_transform(pls, xva_emb)
                s_te = pls_transform(pls, xte_emb)
                xtr = np.concatenate([xtr_emb, s_tr], axis=1)
                xva = np.concatenate([xva_emb, s_va], axis=1)
                xte = np.concatenate([xte_emb, s_te], axis=1)
            else:
                xtr, xva, xte = xtr_emb, xva_emb, xte_emb

            models: List[xgb.Booster] = []
            for j in range(n_targets):
                m = fit_xgb_one(
                    x_tr=xtr,
                    y_tr=ytr_log[:, j],
                    x_va=xva,
                    y_va=yva_log[:, j],
                    xgb_params=xgb_params,
                    seed_value=seed + 1000 * fold_idx + j,
                )
                models.append(m)

            va_log_pred = predict_xgb(models, xva)
            va_pred = np.maximum(np.expm1(va_log_pred), 0.0).astype(np.float32)

            te_log_pred = predict_xgb(models, xte)
            te_pred = np.maximum(np.expm1(te_log_pred), 0.0).astype(np.float32)

            if enforce_constraints:
                va_pred = enforce_sum_constraints(va_pred)
                te_pred = enforce_sum_constraints(te_pred)

            oof_pred[cfg.key][idx_va] = va_pred
            te_sum[cfg.key] += te_pred

            print(f"[{cfg.key}] fold rmse:")
            print(rmse_per_target(true_va, va_pred).to_string())

            del models, xtr, xva, xte
            gc.collect()

    te_bag = {k: v / float(n_folds) for k, v in te_sum.items()}

    print("\n=== oof rmse per model (overall) ===")
    for cfg in model_zoo:
        print(f"[{cfg.key}]")
        print(rmse_per_target(y_clip, oof_pred[cfg.key]).to_string())

    print("\n=== fold-wise oof stack rmse (nnls weights fit excluding fold) ===")
    oof_stack = np.zeros((len(train_images), n_targets), dtype=np.float32)
    model_keys = [cfg.key for cfg in model_zoo]

    for fold_idx in range(1, n_folds + 1):
        tr_mask = fold_id != fold_idx
        va_mask = fold_id == fold_idx
        if not np.any(va_mask):
            continue

        pred_va = np.zeros((va_mask.sum(), n_targets), dtype=np.float32)

        for j in range(n_targets):
            p_tr = np.column_stack([oof_pred[k][tr_mask, j] for k in model_keys])
            y_tr = y_clip[tr_mask, j]
            w = fit_stack_weights_nnls(p_tr, y_tr)

            p_va = np.column_stack([oof_pred[k][va_mask, j] for k in model_keys])
            pred_va[:, j] = (p_va * w.reshape(1, -1)).sum(axis=1)

        pred_va = np.maximum(pred_va, 0.0).astype(np.float32)
        if enforce_constraints:
            pred_va = enforce_sum_constraints(pred_va)

        oof_stack[va_mask] = pred_va

        print(f"\nfold {fold_idx} stack rmse:")
        print(rmse_per_target(y_clip[va_mask], pred_va).to_string())

    print("\noof rmse (stack overall):")
    print(rmse_per_target(y_clip, oof_stack).to_string())

    final_te = np.zeros((len(test_images), n_targets), dtype=np.float32)

    print("\n=== final stack weights per target (fit on all oof) ===")
    for j, t in enumerate(targets):
        p_all = np.column_stack([oof_pred[k][:, j] for k in model_keys])
        w = fit_stack_weights_nnls(p_all, y_clip[:, j])

        w_df = pd.DataFrame({"model": model_keys, "weight": w})
        print(f"\ntarget: {t}")
        print(w_df.sort_values("weight", ascending=False).to_string(index=False))

        p_te = np.column_stack([te_bag[k][:, j] for k in model_keys])
        final_te[:, j] = (p_te * w.reshape(1, -1)).sum(axis=1)

    final_te = np.maximum(final_te, 0.0).astype(np.float32)
    if enforce_constraints:
        final_te = enforce_sum_constraints(final_te)

    pred_wide5 = pd.DataFrame({"image_path": test_images})
    for j, t in enumerate(targets):
        pred_wide5[t] = final_te[:, j]

    build_submission(test_long, pred_wide5[["image_path"] + targets], submission_path)
    print("\nwrote:", submission_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None,
                   help="folder containing train.csv, test.csv, and images referenced by image_path")
    p.add_argument("--weights_dir", type=str, default="weights",
                   help="folder containing your .safetensors files (or set absolute paths in build_default_model_zoo)")
    p.add_argument("--artifacts_dir", type=str, default="artifacts_py",
                   help="cache dir for embeddings")
    p.add_argument("--submission_path", type=str, default="submission.csv",
                   help="output submission csv path")
    p.add_argument("--enforce_constraints", action="store_true", default=True,
                   help="enforce gdm=green+clover and total=green+dead+clover")
    p.add_argument("--no_enforce_constraints", action="store_false", dest="enforce_constraints")
    p.add_argument("--offline", action="store_true", default=True,
                   help="set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1")
    p.add_argument("--online", action="store_false", dest="offline")
    p.add_argument("--nthread", type=int, default=max(1, (os.cpu_count() or 2)),
                   help="xgboost nthread")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_dir=args.data_dir,
        weights_dir=args.weights_dir,
        artifacts_dir=args.artifacts_dir,
        submission_path=args.submission_path,
        enforce_constraints=args.enforce_constraints,
        offline=args.offline,
        nthread=args.nthread,
    )
