# ensemble_biomass_oof_5targets.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import hashlib
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression

from scipy.optimize import nnls

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow import keras
from tensorflow.keras import layers

import timm
import timm.data
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import numpy as np
from PIL import Image

import xgboost as xgb


# =========================
# Config
# =========================
SEED = 123

IMG_SIZE = 224
BATCH_SIZE = 16

N_FOLDS = 5

# Competition requires all 5 targets
TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
N_TARGETS = len(TARGETS)

EPOCHS_PER_FOLD = 10
EPOCHS_FULL = 15
LR = 1e-4

# DINO (timm)
DINO_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
DINO_BATCH = 32

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
    "nthread" : 1
}

# Optional: supervised low-dim features from embeddings (still no mismatch: embeddings exist for train+test)
USE_PLS_ON_DINO = True
PLS_NCOMP = 16

# Cache
CACHE_DIRNAME = "artifacts_py"
DINO_CACHE_BASENAME = "dino_all"  # creates dino_all.npy and dino_all.manifest.json


# =========================
# Repro / stability
# =========================
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_tf_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except (RuntimeError, ValueError):
        return


# =========================
# Metrics (OOF RMSE)
# =========================
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


# =========================
# Data: long -> image-level wide
# =========================
def read_train_image_level(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    wide = (df[["image_path", "target_name", "target"]]
            .drop_duplicates()
            .pivot(index="image_path", columns="target_name", values="target")
            .reset_index())

    # Ensure all 5 exist
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


# =========================
# Safer DINO caching (manifest)
# =========================
def _hash_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def dino_manifest(paths: List[str], model_name: str) -> dict:
    return {
        "model_name": model_name,
        "num_paths": len(paths),
        "paths_hash": _hash_paths(paths),
        "paths_preview": paths[:5],
    }


def load_dino_cache(cache_base: str, paths: List[str], model_name: str) -> Optional[np.ndarray]:
    npy_path = cache_base + ".npy"
    man_path = cache_base + ".manifest.json"
    if not (os.path.exists(npy_path) and os.path.exists(man_path)):
        return None

    with open(man_path, "r") as f:
        man = json.load(f)

    want = dino_manifest(paths, model_name)
    if man.get("model_name") != want["model_name"]:
        return None
    if man.get("paths_hash") != want["paths_hash"]:
        return None

    arr = np.load(npy_path)
    if arr.shape[0] != len(paths):
        return None
    return arr


def save_dino_cache(cache_base: str, arr: np.ndarray, paths: List[str], model_name: str):
    npy_path = cache_base + ".npy"
    man_path = cache_base + ".manifest.json"
    np.save(npy_path, arr)
    with open(man_path, "w") as f:
        json.dump(dino_manifest(paths, model_name), f, indent=2)


# =========================
# DINO embedding
# =========================
@torch.no_grad()
def dinov3_embed_timm(
    image_paths,
    img_root: str,
    model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)

    # IMPORTANT: use model-specific config (input size + normalization)
    data_config = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**data_config, is_training=False)

    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            ap = os.path.join(img_root, p)
            img = Image.open(ap).convert("RGB")
            imgs.append(tfm(img))  # tensor CHW
        x = torch.stack(imgs).to(device)
        f = model(x)  # [B, D]
        feats.append(f.detach().cpu().numpy())

    return np.vstack(feats).astype(np.float32)



def get_all_dino_embeddings(all_paths: List[str], img_root: str, cache_dir: str) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_base = os.path.join(cache_dir, DINO_CACHE_BASENAME)

    cached = load_dino_cache(cache_base, all_paths, DINO_MODEL_NAME)
    if cached is not None:
        return cached.astype(np.float32)

    arr = dinov3_embed_timm(all_paths, img_root=img_root, model_name=DINO_MODEL_NAME)
    save_dino_cache(cache_base, arr, all_paths, DINO_MODEL_NAME)
    return arr.astype(np.float32)


# =========================
# CNN model (image-only) -- no train/test mismatch
# =========================
def build_cnn(img_size: int, n_targets: int, lr: float) -> keras.Model:
    img_in = layers.Input(shape=(img_size, img_size, 3), name="image")

    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=img_in
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(n_targets, activation="linear")(x)

    model = keras.Model(inputs=img_in, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mae")
    return model


def make_tf_dataset(image_paths: List[str],
                    y: Optional[np.ndarray],
                    img_root: str,
                    batch_size: int,
                    shuffle: bool,
                    img_size: int) -> tf.data.Dataset:
    abs_paths = [os.path.join(img_root, p) for p in image_paths]

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(abs_paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((abs_paths, y.astype(np.float32)))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(abs_paths), reshuffle_each_iteration=True)

    def _load_img(path):
        b = tf.io.read_file(path)
        img = tf.image.decode_jpeg(b, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        img = keras.applications.efficientnet.preprocess_input(img)
        return img

    if y is None:
        ds = ds.map(lambda p: _load_img(p), num_parallel_calls=1)
    else:
        ds = ds.map(lambda p, yy: (_load_img(p), yy), num_parallel_calls=1)

    return ds.batch(batch_size).prefetch(1)


# =========================
# XGB helpers
# =========================
def fit_xgb_one(X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray,
                seed: int) -> Tuple[xgb.Booster, int]:
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
    best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds()) or booster.num_boosted_rounds())
    return booster, best_iter


def predict_xgb_models(models: List[xgb.Booster], X: np.ndarray) -> np.ndarray:
    d = xgb.DMatrix(X)
    preds = [m.predict(d) for m in models]
    return np.column_stack(preds).astype(np.float32)


# =========================
# PLS on DINO (optional, inside fold to avoid leakage)
# =========================
@dataclass
class PLSFold:
    pls: PLSRegression
    ncomp: int


def fit_pls_on_embeddings(X: np.ndarray, Y: np.ndarray, ncomp: int) -> PLSFold:
    ncomp2 = int(min(ncomp, X.shape[1], max(1, X.shape[0] - 1)))
    pls = PLSRegression(n_components=ncomp2, scale=True)
    pls.fit(X, Y)
    return PLSFold(pls=pls, ncomp=ncomp2)


def pls_transform(pls_fold: PLSFold, X: np.ndarray) -> np.ndarray:
    return pls_fold.pls.transform(X).astype(np.float32)


# =========================
# Blend weights (NNLS) from OOF
# =========================
def fit_blend_weights(p1: np.ndarray, p2: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Force 1D float arrays
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
    p2 = np.asarray(p2, dtype=np.float64).reshape(-1)
    y  = np.asarray(y,  dtype=np.float64).reshape(-1)

    # Keep only finite rows (nnls can't handle NaN/inf)
    m = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(y)
    p1, p2, y = p1[m], p2[m], y[m]

    # Build design matrix with shape (n, 2)
    W = np.column_stack([p1, p2]).astype(np.float64)

    w, _ = nnls(W, y)  # y must be 1D
    if w.sum() <= 0:
        return np.array([0.5, 0.5], dtype=np.float32)
    return (w / w.sum()).astype(np.float32)


# =========================
# Main CV + Train full + Predict
# =========================
def main(data_dir: str, artifacts_dir: Optional[str] = None):
    set_seed(SEED)
    enable_tf_memory_growth()

    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    artifacts_dir = artifacts_dir or os.path.join(data_dir, CACHE_DIRNAME)
    os.makedirs(artifacts_dir, exist_ok=True)

    # ---- Load
    train_wide = read_train_image_level(train_csv)
    test_long, test_images = read_test_images(test_csv)

    # Deterministic image order (helps caching and reproducibility)
    train_wide = train_wide.sort_values("image_path").reset_index(drop=True)
    test_images = sorted(test_images)

    train_images = train_wide["image_path"].tolist()
    Y = train_wide[TARGETS].to_numpy(dtype=np.float32)

    # train/predict in log-space, invert to grams
    Y_clip = np.clip(Y, 0.0, None)
    Y_log = np.log1p(Y_clip)

    # ---- DINO embeddings for ALL images (train+test) in one consistent order
    all_images = train_images + test_images
    dino_all = get_all_dino_embeddings(all_images, img_root=data_dir, cache_dir=artifacts_dir)
    dino_tr = dino_all[:len(train_images)]
    dino_te = dino_all[len(train_images):]

    # ---- CV folds on images (OOF)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_cnn = np.zeros((len(train_images), N_TARGETS), dtype=np.float32)
    oof_xgb = np.zeros((len(train_images), N_TARGETS), dtype=np.float32)

    best_iters_per_target = [[] for _ in range(N_TARGETS)]

    # Track which fold each training row belongs to (for fold-wise blend diagnostics)
    fold_id = np.full(len(train_images), -1, dtype=np.int32)

    for fold, (idx_tr, idx_va) in enumerate(kf.split(train_images), 1):
        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        fold_id[idx_va] = fold

        tr_paths = [train_images[i] for i in idx_tr]
        va_paths = [train_images[i] for i in idx_va]

        Ytr_log = Y_log[idx_tr]
        Yva_log = Y_log[idx_va]

        # ----- CNN (image-only)
        tf.keras.backend.clear_session()
        cnn = build_cnn(img_size=IMG_SIZE, n_targets=N_TARGETS, lr=LR)

        ds_tr = make_tf_dataset(tr_paths, Ytr_log, data_dir, BATCH_SIZE, shuffle=True, img_size=IMG_SIZE)
        ds_va = make_tf_dataset(va_paths, Yva_log, data_dir, BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]
        cnn.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_PER_FOLD, verbose=1, callbacks=callbacks)

        ds_va_pred = make_tf_dataset(va_paths, None, data_dir, BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)
        pred_cnn_va_log = cnn.predict(ds_va_pred, verbose=0)
        pred_cnn_va = np.maximum(np.expm1(pred_cnn_va_log).astype(np.float32), 0.0)

        # Fold RMSE (CNN) on grams
        true_va = Y_clip[idx_va]
        print("Fold RMSE (CNN):")
        print(rmse_per_target(true_va, pred_cnn_va, TARGETS).to_string())

        oof_cnn[idx_va] = pred_cnn_va

        # ----- XGB on DINO (+ optional PLS-on-DINO), trained in log space
        Xtr_emb = dino_tr[idx_tr]
        Xva_emb = dino_tr[idx_va]

        if USE_PLS_ON_DINO:
            pls_fold = fit_pls_on_embeddings(Xtr_emb, Ytr_log, ncomp=PLS_NCOMP)
            S_tr = pls_transform(pls_fold, Xtr_emb)
            S_va = pls_transform(pls_fold, Xva_emb)
            Xtr = np.concatenate([Xtr_emb, S_tr], axis=1)
            Xva = np.concatenate([Xva_emb, S_va], axis=1)
        else:
            Xtr, Xva = Xtr_emb, Xva_emb

        models = []
        for j in range(N_TARGETS):
            m, best_iter = fit_xgb_one(
                Xtr, Ytr_log[:, j],
                Xva, Yva_log[:, j],
                seed=SEED + fold * 100 + j
            )
            models.append(m)
            best_iters_per_target[j].append(best_iter)

        pred_xgb_va_log = predict_xgb_models(models, Xva)
        pred_xgb_va = np.maximum(np.expm1(pred_xgb_va_log).astype(np.float32), 0.0)

        print("Fold RMSE (XGB):")
        print(rmse_per_target(true_va, pred_xgb_va, TARGETS).to_string())

        oof_xgb[idx_va] = pred_xgb_va

    # =========================
    # TRUE OOF blend RMSE per fold
    # (weights fit on OTHER folds only, then applied to this fold)
    # =========================
    print("\n=== Fold-wise OOF Blend RMSE (weights fit excluding each fold) ===")
    oof_blend_foldwise = np.zeros_like(oof_cnn, dtype=np.float32)

    for fold in range(1, N_FOLDS + 1):
        tr_mask = fold_id != fold
        va_mask = fold_id == fold

        if not np.any(va_mask):
            continue

        w_fold = np.zeros((N_TARGETS, 2), dtype=np.float32)
        pred_va_blend = np.zeros((va_mask.sum(), N_TARGETS), dtype=np.float32)

        for j in range(N_TARGETS):
            wj = fit_blend_weights(oof_cnn[tr_mask, j], oof_xgb[tr_mask, j], Y_clip[tr_mask, j])
            w_fold[j] = wj
            pred_va_blend[:, j] = wj[0] * oof_cnn[va_mask, j] + wj[1] * oof_xgb[va_mask, j]

        pred_va_blend = np.maximum(pred_va_blend, 0.0)
        oof_blend_foldwise[va_mask] = pred_va_blend

        print(f"\nFold {fold} RMSE (BLEND, OOF weights):")
        print(rmse_per_target(Y_clip[va_mask], pred_va_blend, TARGETS).to_string())

    print("\nOOF RMSE (BLEND, fold-wise OOF weights overall):")
    print(rmse_per_target(Y_clip, oof_blend_foldwise, TARGETS).to_string())

    # =========================
    # Final blend weights (fit using ALL OOF) + overall diagnostics
    # =========================
    weights = np.zeros((N_TARGETS, 2), dtype=np.float32)
    for j in range(N_TARGETS):
        w = fit_blend_weights(oof_cnn[:, j], oof_xgb[:, j], Y_clip[:, j])
        weights[j] = w

    print("\nBlend weights [w_cnn, w_xgb] per target (fit on all OOF):")
    print(pd.DataFrame(weights, index=TARGETS, columns=["cnn", "xgb"]))

    print("\nOOF RMSE (CNN):")
    print(rmse_per_target(Y_clip, oof_cnn, TARGETS).to_string())

    print("\nOOF RMSE (XGB):")
    print(rmse_per_target(Y_clip, oof_xgb, TARGETS).to_string())

    oof_blend_allweights = np.zeros_like(oof_cnn, dtype=np.float32)
    for j in range(N_TARGETS):
        oof_blend_allweights[:, j] = weights[j, 0] * oof_cnn[:, j] + weights[j, 1] * oof_xgb[:, j]
    oof_blend_allweights = np.maximum(oof_blend_allweights, 0.0)

    print("\nOOF RMSE (BLEND, weights fit on all OOF):")
    print(rmse_per_target(Y_clip, oof_blend_allweights, TARGETS).to_string())

    # =========================
    # Train FULL models and predict test
    # =========================

    # ---- CNN full
    tf.keras.backend.clear_session()
    cnn_full = build_cnn(img_size=IMG_SIZE, n_targets=N_TARGETS, lr=LR)
    ds_full = make_tf_dataset(train_images, Y_log, data_dir, BATCH_SIZE, shuffle=True, img_size=IMG_SIZE)
    cnn_full.fit(ds_full, epochs=EPOCHS_FULL, verbose=1)

    ds_te = make_tf_dataset(test_images, None, data_dir, BATCH_SIZE, shuffle=False, img_size=IMG_SIZE)
    pred_cnn_te_log = cnn_full.predict(ds_te, verbose=0)
    pred_cnn_te = np.maximum(np.expm1(pred_cnn_te_log).astype(np.float32), 0.0)

    # ---- XGB full: use median best_iteration per target from CV
    if USE_PLS_ON_DINO:
        pls_full = fit_pls_on_embeddings(dino_tr, Y_log, ncomp=PLS_NCOMP)
        S_tr = pls_transform(pls_full, dino_tr)
        S_te = pls_transform(pls_full, dino_te)
        X_tr_full = np.concatenate([dino_tr, S_tr], axis=1)
        X_te_full = np.concatenate([dino_te, S_te], axis=1)
    else:
        X_tr_full, X_te_full = dino_tr, dino_te

    xgb_models_full = []
    for j in range(N_TARGETS):
        rounds = int(np.median(best_iters_per_target[j])) if best_iters_per_target[j] else 2000
        dtr = xgb.DMatrix(X_tr_full, label=Y_log[:, j])
        params = dict(XGB_PARAMS)
        params["seed"] = SEED + 999 + j
        m = xgb.train(params=params, dtrain=dtr, num_boost_round=rounds, verbose_eval=False)
        xgb_models_full.append(m)

    pred_xgb_te_log = predict_xgb_models(xgb_models_full, X_te_full)
    pred_xgb_te = np.maximum(np.expm1(pred_xgb_te_log).astype(np.float32), 0.0)

    # ---- Blend test predictions using final weights (fit on all OOF)
    blended_te = np.zeros_like(pred_cnn_te, dtype=np.float32)
    for j in range(N_TARGETS):
        blended_te[:, j] = weights[j, 0] * pred_cnn_te[:, j] + weights[j, 1] * pred_xgb_te[:, j]
    blended_te = np.maximum(blended_te, 0.0)

    pred_wide5 = pd.DataFrame({"image_path": test_images})
    for j, t in enumerate(TARGETS):
        pred_wide5[t] = blended_te[:, j].astype(np.float32)

    out_path = os.path.join(artifacts_dir, "submission.csv")
    build_submission(test_long, pred_wide5[["image_path"] + TARGETS], out_path)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Folder containing train.csv, test.csv, train/ and test/ image folders")
    parser.add_argument("--artifacts_dir", type=str, default=None, help="Cache + output folder")
    args = parser.parse_args()
    main(args.data_dir, args.artifacts_dir)
