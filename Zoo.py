# model_zoo_v1_kaggle.py
# 4 backbones -> multi-crop embeddings -> XGB per target -> OOF stacking (NNLS) -> submission.csv

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import hashlib
import random
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

from pathlib import Path
import glob


# =========================
# Config
# =========================
SEED = 123
N_FOLDS = 5

NTHREAD = max(1, os.cpu_count() // 1)

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
N_TARGETS = len(TARGETS)

IDX_GREEN  = TARGETS.index("Dry_Green_g")
IDX_DEAD   = TARGETS.index("Dry_Dead_g")
IDX_CLOVER = TARGETS.index("Dry_Clover_g")
IDX_GDM    = TARGETS.index("GDM_g")
IDX_TOTAL  = TARGETS.index("Dry_Total_g")

PANORAMA_N_CROPS = 5  # 3 to start (faster). Try 5 later.

# XGB
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "nthread" : NTHREAD,
    "seed": SEED,
    "tree_method": "hist",
    "max_bin": 256,
}

# Optional PLS (often helps a bit; do it inside-fold)
USE_PLS_ON_EMB = True
PLS_NCOMP = 16

DEFAULT_ARTIFACTS_DIR = "/kaggle/working/artifacts_py"
DEFAULT_SUBMISSION_PATH = "/kaggle/working/submission.csv"
CACHE_BASENAME = "emb_all"  # per-model cache

# ---- 4-model zoo (timm model + local weights path)
@dataclass(frozen=True)
class ModelCfg:
    key: str                  # short id used in cache filenames
    timm_name: str
    weights_path: str
    batch: int = 16
    n_crops: int = PANORAMA_N_CROPS


MODEL_ZOO: List[ModelCfg] = [
    ModelCfg(
        "dinov3b",
        "vit_base_patch16_dinov3.lvd1689m",
        "/kaggle/input/zoolib2-0/model.safetensors",
        batch=16,
    ),

    # ConvNeXtV2 base (matches your weight file)
    ModelCfg(
        "cnxv2t",
        "convnextv2_base.fcmae",
        "/kaggle/input/zoolib2-0/convnextv2_base.model.safetensors",
        batch=16,  # start safer w/ multi-crop
    ),

    # SwinV2-CR base (match the .sw_in1k variant)
    ModelCfg(
        "swincrs",
        "swinv2_base_window16_256.ms_in1k",
        "/kaggle/input/zoolib2-0/swinv2_base_window16_256.model.safetensors",
        batch=12,  # often heavier than convnext
    ),

    # EVA02 Base CLIP (match the merged2b variant)
    ModelCfg(
        "eva02b",
        "eva02_base_patch16_clip_224.merged2b",
        "/kaggle/input/zoolib2-0/eva02_base_patch16_clip_224.merged2b.model.safetensors",
        batch=12,
    ),
]


# =========================
# Repro
# =========================
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Metrics
# =========================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    out = {t: rmse(y_true[:, j], y_pred[:, j]) for j, t in enumerate(TARGETS)}
    out["mean"] = float(np.mean(list(out.values())))
    out["all_concat"] = rmse(y_true.reshape(-1), y_pred.reshape(-1))
    return pd.Series(out)


# =========================
# Data I/O
# =========================
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


def find_kaggle_data_dir() -> Optional[str]:
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    for root, _, files in os.walk(base):
        if "train.csv" in files and "test.csv" in files:
            return root
    return None


# =========================
# Weights loader (local only)
# =========================
def load_local_weights_into_timm(model, weights_path: str) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

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
    print(f"[weights] loaded {weights_path} | missing={len(missing)} unexpected={len(unexpected)}")

def resolve_weights_path(path_or_dir: str, prefer_contains: str | None = None) -> str:
    p = Path(path_or_dir)

    # exact file path
    if p.is_file():
        return str(p)

    # directory -> pick a safetensors file inside it
    if p.is_dir():
        cands = sorted(glob.glob(str(p / "**" / "*.safetensors"), recursive=True))
        if not cands:
            raise FileNotFoundError(f"No .safetensors found under: {p}")
        if prefer_contains:
            for c in cands:
                if prefer_contains in Path(c).name:
                    return c
        return cands[0]

    raise FileNotFoundError(f"Missing weights path/dir: {path_or_dir}")

def _weights_fingerprint(path: str) -> str:
    # fast + good enough: size + mtime
    st = os.stat(path)
    return f"{st.st_size}:{int(st.st_mtime)}"


# =========================
# Caching
# =========================
def _hash_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def emb_manifest(paths: List[str], cfg: ModelCfg) -> dict:
    return {
        "timm_name": cfg.timm_name,
        "key": cfg.key,
        "n_crops": int(cfg.n_crops),
        "paths_hash": _hash_paths(paths),
        "weights_fp": _weights_fingerprint(cfg.weights_path),
        "num_paths": len(paths),
        "preview": paths[:5],
    }

def load_emb_cache(cache_base: str, paths: List[str], cfg: ModelCfg) -> Optional[np.ndarray]:
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

def save_emb_cache(cache_base: str, arr: np.ndarray, paths: List[str], cfg: ModelCfg):
    np.save(cache_base + ".npy", arr.astype(np.float32))
    with open(cache_base + ".manifest.json", "w") as f:
        json.dump(emb_manifest(paths, cfg), f, indent=2)


# =========================
# Panorama crops
# =========================
def panorama_square_crops(img: Image.Image, n_crops: int) -> List[Image.Image]:
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


# =========================
# Embeddings (generic multi-crop mean pool)
# =========================
@torch.inference_mode()
def embed_timm_multicrop_mean(
    image_paths: List[str],
    img_root: str,
    cfg: ModelCfg,
    device: Optional[str] = None,
) -> np.ndarray:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(cfg.timm_name, pretrained=False, num_classes=0)
    wpath = cfg.weights_path
    load_local_weights_into_timm(model, wpath)
    model.eval().to(device)

    data_config = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**data_config, is_training=False)
    use_amp = device.startswith("cuda")

    feats = []
    bs = int(cfg.batch)

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

        f = f.view(B, V, -1).mean(dim=1)  # [B,D]
        feats.append(f.float().cpu().numpy())

        if use_amp:
            del x, f
            torch.cuda.empty_cache()

    return np.vstack(feats).astype(np.float32)


def get_all_embeddings(all_paths: List[str], img_root: str, cache_dir: str, cfg: ModelCfg) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_base = os.path.join(cache_dir, f"{CACHE_BASENAME}_{cfg.key}_c{cfg.n_crops}")
    cached = load_emb_cache(cache_base, all_paths, cfg)
    if cached is not None:
        print(f"[cache] hit: {cfg.key}")
        return cached
    print(f"[cache] miss: {cfg.key} -> computing embeddings...")
    arr = embed_timm_multicrop_mean(all_paths, img_root=img_root, cfg=cfg)
    save_emb_cache(cache_base, arr, all_paths, cfg)
    return arr


# =========================
# XGB
# =========================
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

def predict_xgb(models: List[xgb.Booster], X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    cols = []
    for m in models:
        if hasattr(m, "best_iteration") and m.best_iteration is not None:
            cols.append(m.predict(d, iteration_range=(0, m.best_iteration + 1)))
        else:
            cols.append(m.predict(d))
    return np.column_stack(cols).astype(np.float32)


# =========================
# PLS (optional)
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


# =========================
# Optional: enforce known identities at inference
# =========================
def enforce_sum_constraints(pred5: np.ndarray) -> np.ndarray:
    pred5 = np.asarray(pred5, dtype=np.float32).copy()
    pred5 = np.maximum(pred5, 0.0)
    g = pred5[:, IDX_GREEN]
    d = pred5[:, IDX_DEAD]
    c = pred5[:, IDX_CLOVER]
    pred5[:, IDX_GDM] = g + c
    pred5[:, IDX_TOTAL] = g + d + c
    return pred5


# =========================
# Stacking weights (NNLS, K models)
# =========================
def fit_stack_weights_nnls(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    # P shape: (n_samples, n_models); y shape: (n_samples,)
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    m = np.isfinite(y)
    m = m & np.all(np.isfinite(P), axis=1)
    P = P[m]
    y = y[m]

    w, _ = nnls(P, y)
    if w.sum() <= 0:
        w = np.ones(P.shape[1], dtype=np.float64) / P.shape[1]
    else:
        w = w / w.sum()
    return w.astype(np.float32)


# =========================
# Main
# =========================
def main(data_dir: Optional[str] = None,
         artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
         submission_path: str = DEFAULT_SUBMISSION_PATH,
         enforce_constraints: bool = True):

    set_seed(SEED)

    if data_dir is None:
        data_dir = find_kaggle_data_dir()
    if data_dir is None:
        raise FileNotFoundError("Could not auto-detect data dir under /kaggle/input")

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    os.makedirs(artifacts_dir, exist_ok=True)

    train_wide = read_train_image_level(train_csv).sort_values("image_path").reset_index(drop=True)
    test_long, test_images = read_test_images(test_csv)
    test_images = sorted(test_images)

    train_images = train_wide["image_path"].tolist()
    Y = train_wide[TARGETS].to_numpy(dtype=np.float32)
    Y_clip = np.clip(Y, 0.0, None)
    Y_log = np.log1p(Y_clip)

    print(f"Train: {len(train_images)} | Test: {len(test_images)} | Models: {len(MODEL_ZOO)}")
    for cfg in MODEL_ZOO:
        print(" -", cfg.key, cfg.timm_name, "| crops", cfg.n_crops, "| batch", cfg.batch)

    # Resolve all weight paths once (ensures cache fp matches what gets loaded)
    resolved_zoo = []
    for cfg in MODEL_ZOO:
        wpath = resolve_weights_path(cfg.weights_path, prefer_contains=cfg.timm_name)
        resolved_zoo.append(ModelCfg(cfg.key, cfg.timm_name, wpath, batch=cfg.batch, n_crops=cfg.n_crops))
    MODEL_ZOO[:] = resolved_zoo


    # ---- embeddings for all models (train+test)
    all_images = train_images + test_images
    emb_tr: Dict[str, np.ndarray] = {}
    emb_te: Dict[str, np.ndarray] = {}

    for cfg in MODEL_ZOO:
        arr = get_all_embeddings(all_images, img_root=data_dir, cache_dir=artifacts_dir, cfg=cfg)
        emb_tr[cfg.key] = arr[:len(train_images)]
        emb_te[cfg.key] = arr[len(train_images):]
        del arr
        gc.collect()

    # ---- CV
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # store OOF preds per model: dict key -> (n_train, 5)
    oof_pred: Dict[str, np.ndarray] = {cfg.key: np.zeros((len(train_images), N_TARGETS), np.float32) for cfg in MODEL_ZOO}
    # store bagged test preds per model
    te_sum: Dict[str, np.ndarray] = {cfg.key: np.zeros((len(test_images), N_TARGETS), np.float32) for cfg in MODEL_ZOO}
    fold_id = np.full(len(train_images), -1, dtype=np.int32)

    for fold, (idx_tr, idx_va) in enumerate(kf.split(train_images), 1):
        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        fold_id[idx_va] = fold

        Ytr_log, Yva_log = Y_log[idx_tr], Y_log[idx_va]
        true_va = Y_clip[idx_va]

        # train each model's XGB and produce preds
        for cfg in MODEL_ZOO:
            Xtr_emb = emb_tr[cfg.key][idx_tr]
            Xva_emb = emb_tr[cfg.key][idx_va]
            Xte_emb = emb_te[cfg.key]

            if USE_PLS_ON_EMB:
                pls = fit_pls_on_embeddings(Xtr_emb, Ytr_log, ncomp=PLS_NCOMP)
                S_tr = pls_transform(pls, Xtr_emb)
                S_va = pls_transform(pls, Xva_emb)
                S_te = pls_transform(pls, Xte_emb)
                Xtr = np.concatenate([Xtr_emb, S_tr], axis=1)
                Xva = np.concatenate([Xva_emb, S_va], axis=1)
                Xte = np.concatenate([Xte_emb, S_te], axis=1)
            else:
                Xtr, Xva, Xte = Xtr_emb, Xva_emb, Xte_emb

            models = []
            for j in range(N_TARGETS):
                m = fit_xgb_one(Xtr, Ytr_log[:, j], Xva, Yva_log[:, j], seed=SEED + 1000*fold + j)
                models.append(m)

            va_log_pred = predict_xgb(models, Xva)
            va_pred = np.maximum(np.expm1(va_log_pred), 0.0).astype(np.float32)

            te_log_pred = predict_xgb(models, Xte)
            te_pred = np.maximum(np.expm1(te_log_pred), 0.0).astype(np.float32)

            if enforce_constraints:
                va_pred = enforce_sum_constraints(va_pred)
                te_pred = enforce_sum_constraints(te_pred)

            oof_pred[cfg.key][idx_va] = va_pred
            te_sum[cfg.key] += te_pred

            print(f"[{cfg.key}] Fold RMSE:")
            print(rmse_per_target(true_va, va_pred).to_string())

            del models, Xtr, Xva, Xte
            gc.collect()

    # bagged test preds per model
    te_bag = {k: v / float(N_FOLDS) for k, v in te_sum.items()}

    # overall OOF per-model
    print("\n=== OOF RMSE per model (overall) ===")
    for cfg in MODEL_ZOO:
        print(f"[{cfg.key}]")
        print(rmse_per_target(Y_clip, oof_pred[cfg.key]).to_string())

    # =========================
    # Fold-wise OOF stacking RMSE (weights fit excluding fold)
    # =========================
    print("\n=== Fold-wise OOF STACK RMSE (NNLS weights fit excluding fold) ===")
    oof_stack = np.zeros((len(train_images), N_TARGETS), dtype=np.float32)

    model_keys = [cfg.key for cfg in MODEL_ZOO]
    K = len(model_keys)

    for fold in range(1, N_FOLDS + 1):
        tr_mask = fold_id != fold
        va_mask = fold_id == fold
        if not np.any(va_mask):
            continue

        pred_va = np.zeros((va_mask.sum(), N_TARGETS), dtype=np.float32)

        for j in range(N_TARGETS):
            P_tr = np.column_stack([oof_pred[k][tr_mask, j] for k in model_keys])  # (n_tr, K)
            y_tr = Y_clip[tr_mask, j]
            w = fit_stack_weights_nnls(P_tr, y_tr)

            P_va = np.column_stack([oof_pred[k][va_mask, j] for k in model_keys])  # (n_va, K)
            pred_va[:, j] = (P_va * w.reshape(1, -1)).sum(axis=1)

        pred_va = np.maximum(pred_va, 0.0).astype(np.float32)
        if enforce_constraints:
            pred_va = enforce_sum_constraints(pred_va)

        oof_stack[va_mask] = pred_va

        print(f"\nFold {fold} STACK RMSE:")
        print(rmse_per_target(Y_clip[va_mask], pred_va).to_string())

    print("\nOOF RMSE (STACK overall):")
    print(rmse_per_target(Y_clip, oof_stack).to_string())

    # =========================
    # Final stack weights (fit on all OOF) + apply to bagged test preds
    # =========================
    final_te = np.zeros((len(test_images), N_TARGETS), dtype=np.float32)

    print("\n=== Final stack weights per target (fit on all OOF) ===")
    for j, t in enumerate(TARGETS):
        P_all = np.column_stack([oof_pred[k][:, j] for k in model_keys])
        w = fit_stack_weights_nnls(P_all, Y_clip[:, j])

        w_df = pd.DataFrame({"model": model_keys, "weight": w})
        print(f"\nTarget: {t}")
        print(w_df.sort_values("weight", ascending=False).to_string(index=False))

        P_te = np.column_stack([te_bag[k][:, j] for k in model_keys])
        final_te[:, j] = (P_te * w.reshape(1, -1)).sum(axis=1)

    final_te = np.maximum(final_te, 0.0).astype(np.float32)
    if enforce_constraints:
        final_te = enforce_sum_constraints(final_te)

    # write submission
    pred_wide5 = pd.DataFrame({"image_path": test_images})
    for j, t in enumerate(TARGETS):
        pred_wide5[t] = final_te[:, j]

    build_submission(test_long, pred_wide5[["image_path"] + TARGETS], submission_path)
    print("\nWrote:", submission_path)


if __name__ == "__main__":
    main()