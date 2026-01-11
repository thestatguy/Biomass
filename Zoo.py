# model_zoo_v1_stack_xgb.py
# Kaggle-ready: multi-backbone embeddings -> XGB OOF -> NNLS stacking -> submission.csv

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc, json, hashlib, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from scipy.optimize import nnls

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import timm
import timm.data
from PIL import Image

import xgboost as xgb


# =========================
# Config
# =========================
SEED = 123

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
N_TARGETS = len(TARGETS)

IDX_GREEN  = TARGETS.index("Dry_Green_g")
IDX_DEAD   = TARGETS.index("Dry_Dead_g")
IDX_CLOVER = TARGETS.index("Dry_Clover_g")
IDX_GDM    = TARGETS.index("GDM_g")
IDX_TOTAL  = TARGETS.index("Dry_Total_g")

N_FOLDS = 5

# Panorama crops (2000x1000-ish)
PANORAMA_N_CROPS_DEFAULT = 3

# XGB
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": SEED,
    "nthread": 1,
}

# Optional: PLS on embeddings INSIDE each fold (leak-safe)
USE_PLS_ON_EMB = True
PLS_NCOMP = 16

# Optional: enforce exact target identities at inference
ENFORCE_SUMS = True

# Cache / outputs
DEFAULT_ARTIFACTS_DIR = "/kaggle/working/artifacts_py"
DEFAULT_SUBMISSION_PATH = "/kaggle/working/submission.csv"
DINO_CACHE_DIRNAME = "emb_cache"  # per-model cache inside artifacts


# =========================
# MODEL ZOO v1 (EDIT THIS!)
# =========================
@dataclass
class BackboneCfg:
    tag: str
    model_name: str
    weights_path: str  # local file in /kaggle/input/...
    n_crops: int = PANORAMA_N_CROPS_DEFAULT
    batch_size: int = 16

# IMPORTANT:
# 1) Put the REAL timm model names (print candidates using the helper below).
# 2) Upload each backbone weights file as a Kaggle Dataset and set weights_path accordingly.
MODEL_ZOO: List[BackboneCfg] = [
    BackboneCfg(
        tag="dinov3_b16",
        model_name="vit_base_patch16_dinov3.lvd1689m",
        weights_path="/kaggle/input/dinov3/model.safetensors",
        n_crops=3,
        batch_size=16,
    ),
    # EXAMPLES (you must replace model_name + weights_path with valid ones in your runtime):
    # BackboneCfg(tag="convnextv2_base", model_name="convnextv2_base.<...>", weights_path="/kaggle/input/convnextv2/model.safetensors", n_crops=3, batch_size=16),
    # BackboneCfg(tag="swinv2_base",     model_name="swinv2_base_window12_192.<...>", weights_path="/kaggle/input/swinv2/model.safetensors", n_crops=3, batch_size=16),
    # BackboneCfg(tag="clip_vit_b16",    model_name="vit_base_patch16_clip_224.<...>", weights_path="/kaggle/input/clip/model.safetensors", n_crops=3, batch_size=16),
]


# =========================
# Helpers
# =========================
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> pd.Series:
    out = {}
    for j, t in enumerate(target_names):
        out[t] = rmse(y_true[:, j], y_pred[:, j])
    out["mean"] = float(np.mean(list(out.values())))
    out["all_concat"] = rmse(y_true.reshape(-1), y_pred.reshape(-1))
    return pd.Series(out)

def find_kaggle_data_dir() -> Optional[str]:
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    for root, _, files in os.walk(base):
        fs = set(files)
        if "train.csv" in fs and "test.csv" in fs:
            return root
    return None

def read_train_image_level(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    wide = (df[["image_path", "target_name", "target"]]
            .drop_duplicates()
            .pivot(index="image_path", columns="target_name", values="target")
            .reset_index())
    for t in TARGETS:
        if t not in wide.columns:
            wide[t] = np.nan
    return wide

def read_test_images(test_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    test_long = pd.read_csv(test_csv)
    test_images = test_long["image_path"].drop_duplicates().tolist()
    return test_long, test_images

def build_submission(test_long: pd.DataFrame, pred_wide5: pd.DataFrame, out_path: str):
    pred_long = pred_wide5.melt(
        id_vars=["image_path"],
        value_vars=TARGETS,
        var_name="target_name",
        value_name="target",
    )
    sub = (test_long
           .merge(pred_long, on=["image_path", "target_name"], how="left")
           [["sample_id", "target"]])
    sub.to_csv(out_path, index=False)

def enforce_sum_constraints(pred5: np.ndarray) -> np.ndarray:
    pred5 = np.asarray(pred5, dtype=np.float32).copy()
    pred5 = np.maximum(pred5, 0.0)
    green  = pred5[:, IDX_GREEN]
    dead   = pred5[:, IDX_DEAD]
    clover = pred5[:, IDX_CLOVER]
    pred5[:, IDX_GDM]   = green + clover
    pred5[:, IDX_TOTAL] = green + dead + clover
    return pred5

def print_timm_candidates():
    # Run this once to see what timm names exist in your environment.
    pats = ["*dinov2*", "*dinov3*", "*convnextv2*", "*swinv2*", "*clip*"]
    for p in pats:
        ms = timm.list_models(p)
        print(p, "->", len(ms))
        print("  ", ms[:10])


# =========================
# Local weights loading (offline)
# =========================
def load_local_weights_into_timm(model, weights_path: str) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights_path)
    else:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        kk = k
        for pref in ("model.", "module.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[weights] {os.path.basename(weights_path)} | missing={len(missing)} unexpected={len(unexpected)}")


# =========================
# Embedding cache
# =========================
def _hash_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def _weights_sig(path: str) -> dict:
    st = os.stat(path)
    return {"path": path, "size": int(st.st_size), "mtime": int(st.st_mtime)}

def emb_manifest(paths: List[str], cfg: BackboneCfg) -> dict:
    return {
        "tag": cfg.tag,
        "model_name": cfg.model_name,
        "n_crops": int(cfg.n_crops),
        "batch_size": int(cfg.batch_size),
        "paths_hash": _hash_paths(paths),
        "num_paths": len(paths),
        "weights": _weights_sig(cfg.weights_path),
    }

def load_emb_cache(cache_base: str, paths: List[str], cfg: BackboneCfg) -> Optional[np.ndarray]:
    npy_path = cache_base + ".npy"
    man_path = cache_base + ".manifest.json"
    if not (os.path.exists(npy_path) and os.path.exists(man_path)):
        return None
    with open(man_path, "r") as f:
        man = json.load(f)
    want = emb_manifest(paths, cfg)

    # strict checks to avoid mismatched embeddings
    for k in ("tag", "model_name", "n_crops", "batch_size", "paths_hash", "num_paths"):
        if man.get(k) != want.get(k):
            return None
    # weights signature match
    if man.get("weights", {}) != want.get("weights", {}):
        return None

    arr = np.load(npy_path)
    if arr.shape[0] != len(paths):
        return None
    return arr.astype(np.float32)

def save_emb_cache(cache_base: str, arr: np.ndarray, paths: List[str], cfg: BackboneCfg):
    np.save(cache_base + ".npy", arr.astype(np.float32))
    with open(cache_base + ".manifest.json", "w") as f:
        json.dump(emb_manifest(paths, cfg), f, indent=2)


# =========================
# Panorama crops + embedding extraction
# =========================
def panorama_square_crops(img: Image.Image, n_crops: int = 3) -> List[Image.Image]:
    w, h = img.size
    s = min(w, h)  # for 2000x1000 -> 1000
    if w <= s:
        return [img]
    max_left = w - s
    if n_crops <= 1:
        lefts = [max_left // 2]
    else:
        lefts = [int(round(i * max_left / (n_crops - 1))) for i in range(n_crops)]
    return [img.crop((l, 0, l + s, s)) for l in lefts]

@torch.inference_mode()
def embed_timm_multicrop_mean(
    image_paths: List[str],
    img_root: str,
    cfg: BackboneCfg,
    device: Optional[str] = None,
) -> np.ndarray:

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Build model (offline)
    if cfg.model_name not in timm.list_models(cfg.model_name):
        # exact name not found; give helpful message
        raise RuntimeError(f"Unknown timm model: {cfg.model_name}. Use timm.list_models() to find the exact name.")

    model = timm.create_model(cfg.model_name, pretrained=False, num_classes=0)
    load_local_weights_into_timm(model, cfg.weights_path)
    model.eval().to(device)

    data_config = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**data_config, is_training=False)

    use_amp = device.startswith("cuda")
    feats = []

    bs = int(cfg.batch_size)
    for i in range(0, len(image_paths), bs):
        batch_paths = image_paths[i:i + bs]
        packed = []
        for p in batch_paths:
            ap = os.path.join(img_root, p)
            with Image.open(ap) as im:
                img = im.convert("RGB")
            crops = panorama_square_crops(img, n_crops=cfg.n_crops)
            packed.append(torch.stack([tfm(c) for c in crops], dim=0))  # [V,C,H,W]

        x = torch.stack(packed, dim=0)  # [B,V,C,H,W] CPU
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W).to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                f = model(x)
        else:
            f = model(x)

        f = f.view(B, V, -1).mean(dim=1)      # [B,D]
        feats.append(f.float().cpu().numpy())  # float32

        if use_amp:
            del x, f
            torch.cuda.empty_cache()

    return np.vstack(feats).astype(np.float32)

def get_all_embeddings(all_paths: List[str], img_root: str, artifacts_dir: str, cfg: BackboneCfg) -> np.ndarray:
    cache_dir = os.path.join(artifacts_dir, DINO_CACHE_DIRNAME, cfg.tag)
    os.makedirs(cache_dir, exist_ok=True)
    cache_base = os.path.join(cache_dir, "emb_all")

    cached = load_emb_cache(cache_base, all_paths, cfg)
    if cached is not None:
        print(f"[cache] hit: {cfg.tag} -> {cached.shape}")
        return cached

    print(f"[cache] miss: {cfg.tag} -> computing embeddings...")
    arr = embed_timm_multicrop_mean(all_paths, img_root=img_root, cfg=cfg)
    save_emb_cache(cache_base, arr, all_paths, cfg)
    print(f"[cache] saved: {cfg.tag} -> {arr.shape}")
    return arr


# =========================
# XGB + optional PLS inside fold
# =========================
@dataclass
class PLSFold:
    pls: PLSRegression

def fit_pls_on_embeddings(X: np.ndarray, Y: np.ndarray, ncomp: int) -> PLSFold:
    ncomp2 = int(min(ncomp, X.shape[1], max(1, X.shape[0] - 1)))
    pls = PLSRegression(n_components=ncomp2, scale=True)
    pls.fit(X, Y)
    return PLSFold(pls=pls)

def pls_transform(pls_fold: PLSFold, X: np.ndarray) -> np.ndarray:
    return pls_fold.pls.transform(X).astype(np.float32)

def fit_xgb_one(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, seed: int) -> xgb.Booster:
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    params = dict(XGB_PARAMS)
    params["seed"] = seed
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=5000,
        evals=[(dva, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return booster

def predict_xgb_models(models: List[xgb.Booster], X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    preds = [m.predict(d) for m in models]
    return np.column_stack(preds).astype(np.float32)

def run_xgb_oof_for_backbone(
    tag: str,
    emb_tr: np.ndarray,
    emb_te: np.ndarray,
    Y_log: np.ndarray,
    Y_clip: np.ndarray,
    kf: KFold,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      oof_pred_grams: (n_train, 5)
      te_pred_bag_grams: (n_test, 5)
    """
    n_train = emb_tr.shape[0]
    n_test = emb_te.shape[0]

    oof = np.zeros((n_train, N_TARGETS), dtype=np.float32)
    te_sum = np.zeros((n_test, N_TARGETS), dtype=np.float32)

    for fold, (idx_tr, idx_va) in enumerate(kf.split(np.arange(n_train)), 1):
        print(f"\n[{tag}] Fold {fold}/{kf.get_n_splits()}")

        Xtr_emb = emb_tr[idx_tr]
        Xva_emb = emb_tr[idx_va]
        Ytr_log = Y_log[idx_tr]
        Yva_log = Y_log[idx_va]

        # PLS inside fold (leak-safe)
        if USE_PLS_ON_EMB:
            pls_fold = fit_pls_on_embeddings(Xtr_emb, Ytr_log, ncomp=PLS_NCOMP)
            S_tr = pls_transform(pls_fold, Xtr_emb)
            S_va = pls_transform(pls_fold, Xva_emb)
            Xtr = np.concatenate([Xtr_emb, S_tr], axis=1)
            Xva = np.concatenate([Xva_emb, S_va], axis=1)

            S_te = pls_transform(pls_fold, emb_te)
            Xte = np.concatenate([emb_te, S_te], axis=1)
        else:
            Xtr, Xva, Xte = Xtr_emb, Xva_emb, emb_te

        # fit 5 separate xgbs
        models = []
        for j in range(N_TARGETS):
            m = fit_xgb_one(Xtr, Ytr_log[:, j], Xva, Yva_log[:, j], seed=SEED + 1000*fold + j)
            models.append(m)

        # val preds -> grams
        pred_va_log = predict_xgb_models(models, Xva)
        pred_va = np.maximum(np.expm1(pred_va_log), 0.0).astype(np.float32)
        if ENFORCE_SUMS:
            pred_va = enforce_sum_constraints(pred_va)

        oof[idx_va] = pred_va

        # test preds -> grams (fold-bag)
        pred_te_log = predict_xgb_models(models, Xte)
        pred_te = np.maximum(np.expm1(pred_te_log), 0.0).astype(np.float32)
        if ENFORCE_SUMS:
            pred_te = enforce_sum_constraints(pred_te)

        te_sum += pred_te

        # diagnostics
        print("Fold RMSE (XGB):")
        print(rmse_per_target(Y_clip[idx_va], pred_va, TARGETS).to_string())

        del models
        gc.collect()

    te_bag = te_sum / float(kf.get_n_splits())
    print(f"\n[{tag}] OOF RMSE overall:")
    print(rmse_per_target(Y_clip, oof, TARGETS).to_string())
    return oof, te_bag


# =========================
# NNLS stacking (K models) per target + fold-wise evaluation
# =========================
def nnls_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    P: (n, K) predictions (grams)
    y: (n,) target (grams)
    returns w: (K,) nonneg weights, sum to 1
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    m = np.isfinite(P).all(axis=1) & np.isfinite(y)
    P = P[m]
    y = y[m]

    w, _ = nnls(P, y)
    if w.sum() <= 0:
        w = np.ones(P.shape[1], dtype=np.float64) / P.shape[1]
    else:
        w = w / w.sum()
    return w.astype(np.float32)

def stack_foldwise_oof(
    oof_by_model: Dict[str, np.ndarray],
    Y_clip: np.ndarray,
    fold_id: np.ndarray,
) -> np.ndarray:
    """
    fold-wise evaluation: for each fold f, fit weights on other folds OOF,
    apply to fold f. Returns stacked OOF predictions (n_train, 5).
    """
    model_tags = list(oof_by_model.keys())
    K = len(model_tags)
    n = Y_clip.shape[0]

    stacked = np.zeros((n, N_TARGETS), dtype=np.float32)

    for f in sorted(np.unique(fold_id)):
        tr_mask = fold_id != f
        va_mask = fold_id == f
        if not np.any(va_mask):
            continue

        pred_va = np.zeros((va_mask.sum(), N_TARGETS), dtype=np.float32)

        for j in range(N_TARGETS):
            P_tr = np.column_stack([oof_by_model[tag][tr_mask, j] for tag in model_tags])  # (ntr, K)
            y_tr = Y_clip[tr_mask, j]
            wj = nnls_weights(P_tr, y_tr)

            P_va = np.column_stack([oof_by_model[tag][va_mask, j] for tag in model_tags])  # (nva, K)
            pred_va[:, j] = (P_va @ wj).astype(np.float32)

        pred_va = np.maximum(pred_va, 0.0)
        if ENFORCE_SUMS:
            pred_va = enforce_sum_constraints(pred_va)

        stacked[va_mask] = pred_va
        print(f"\n[STACK] Fold {int(f)} RMSE (NNLS OOF weights):")
        print(rmse_per_target(Y_clip[va_mask], pred_va, TARGETS).to_string())

    print("\n[STACK] OOF RMSE overall (fold-wise OOF weights):")
    print(rmse_per_target(Y_clip, stacked, TARGETS).to_string())
    return stacked

def stack_fit_all_and_predict_test(
    oof_by_model: Dict[str, np.ndarray],
    te_by_model: Dict[str, np.ndarray],
    Y_clip: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame]:
    model_tags = list(oof_by_model.keys())
    K = len(model_tags)

    weights = np.zeros((N_TARGETS, K), dtype=np.float32)

    for j in range(N_TARGETS):
        P = np.column_stack([oof_by_model[tag][:, j] for tag in model_tags])
        y = Y_clip[:, j]
        weights[j] = nnls_weights(P, y)

    wdf = pd.DataFrame(weights, index=TARGETS, columns=model_tags)
    print("\n[STACK] NNLS weights fit on ALL OOF:")
    print(wdf.to_string())

    # test prediction
    n_test = next(iter(te_by_model.values())).shape[0]
    te_pred = np.zeros((n_test, N_TARGETS), dtype=np.float32)

    for j in range(N_TARGETS):
        P_te = np.column_stack([te_by_model[tag][:, j] for tag in model_tags])
        te_pred[:, j] = (P_te @ weights[j]).astype(np.float32)

    te_pred = np.maximum(te_pred, 0.0)
    if ENFORCE_SUMS:
        te_pred = enforce_sum_constraints(te_pred)

    return te_pred, wdf


# =========================
# Main
# =========================
def main(data_dir: Optional[str] = None,
         artifacts_dir: Optional[str] = None,
         submission_path: Optional[str] = None):

    set_seed(SEED)

    if data_dir is None:
        data_dir = find_kaggle_data_dir()
    if data_dir is None:
        raise FileNotFoundError("Could not auto-detect data_dir. Provide it explicitly.")

    artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    submission_path = submission_path or DEFAULT_SUBMISSION_PATH
    os.makedirs(artifacts_dir, exist_ok=True)

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv  = os.path.join(data_dir, "test.csv")

    train_wide = read_train_image_level(train_csv)
    test_long, test_images = read_test_images(test_csv)

    # deterministic ordering (important for caching + reproducibility)
    train_wide = train_wide.sort_values("image_path").reset_index(drop=True)
    test_images = sorted(test_images)

    train_images = train_wide["image_path"].tolist()
    Y = train_wide[TARGETS].to_numpy(dtype=np.float32)
    Y_clip = np.clip(Y, 0.0, None)
    Y_log  = np.log1p(Y_clip)

    print("Data dir:", data_dir)
    print("Artifacts dir:", artifacts_dir)
    print("Train images:", len(train_images), "| Test images:", len(test_images))
    print("Torch CUDA:", torch.cuda.is_available())

    # build folds once (used for all models + fold-wise stacking)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_id = np.full(len(train_images), -1, dtype=np.int32)
    for f, (_, idx_va) in enumerate(kf.split(np.arange(len(train_images))), 1):
        fold_id[idx_va] = f

    # all images for embeddings (train+test)
    all_images = train_images + test_images

    # store per-model preds
    oof_by_model: Dict[str, np.ndarray] = {}
    te_by_model: Dict[str, np.ndarray] = {}

    # loop over zoo
    for cfg in MODEL_ZOO:
        print("\n" + "="*80)
        print(f"[ZOO] {cfg.tag} | {cfg.model_name} | crops={cfg.n_crops} | bs={cfg.batch_size}")
        print("="*80)

        # sanity
        if not os.path.exists(cfg.weights_path):
            raise FileNotFoundError(f"[{cfg.tag}] missing weights file: {cfg.weights_path}")

        # embeddings (cached)
        emb_all = get_all_embeddings(all_images, img_root=data_dir, artifacts_dir=artifacts_dir, cfg=cfg)
        emb_tr = emb_all[:len(train_images)]
        emb_te = emb_all[len(train_images):]

        # xgb OOF + fold-bag test
        oof, te_bag = run_xgb_oof_for_backbone(cfg.tag, emb_tr, emb_te, Y_log, Y_clip, kf)

        oof_by_model[cfg.tag] = oof
        te_by_model[cfg.tag] = te_bag

        # free memory between backbones
        del emb_all, emb_tr, emb_te
        gc.collect()

    # =========================
    # Fold-wise stacking eval (OOF weights excluding each fold)
    # =========================
    if len(oof_by_model) >= 2:
        _ = stack_foldwise_oof(oof_by_model, Y_clip, fold_id)
        te_stack, wdf = stack_fit_all_and_predict_test(oof_by_model, te_by_model, Y_clip)
        final_te = te_stack
        print("\nUsing STACKED predictions for submission.")
    else:
        # only one model in zoo
        tag = next(iter(te_by_model.keys()))
        final_te = te_by_model[tag]
        print(f"\nOnly one model found ({tag}); using it directly for submission.")

    # write submission
    pred_wide5 = pd.DataFrame({"image_path": test_images})
    for j, t in enumerate(TARGETS):
        pred_wide5[t] = final_te[:, j].astype(np.float32)

    build_submission(test_long, pred_wide5[["image_path"] + TARGETS], submission_path)
    print("\nWrote:", submission_path)


if __name__ == "__main__":
    # In a Kaggle Notebook: just run this file; it will auto-detect the dataset folder in /kaggle/input
    # Tip: run print_timm_candidates() once if you need model names.
    main()
