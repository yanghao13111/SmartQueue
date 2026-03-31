"""
SmartQueue Stage B - Personalized Ranking Model Training
Trains a skip/non-skip prediction model using session interaction data.
Supports Logistic Regression (baseline) and LightGBM via YAML config.

Usage:
    python train_ranking.py configs/stage_b_baseline.yaml
    python train_ranking.py configs/stage_b_lgbm_v1.yaml
"""

import os
import sys
import time
import yaml
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_and_prepare_data(cfg: dict) -> pd.DataFrame:
    """
    Load XITE parquet, create engagement label, and build features.
    Uses head(max_samples) to control memory on small VMs.
    """
    print(f"[data] Loading from {cfg['data_path']}...")
    max_samples = cfg.get("max_samples", 50000)
    df = pd.read_parquet(cfg["data_path"]).head(max_samples)
    print(f"[data] Loaded {len(df)} rows")

    # --- Label Engineering ---
    # skip = listened less than threshold seconds
    # engaged (non-skip) = listened >= threshold seconds
    threshold = cfg.get("skip_threshold_seconds", 30)
    df["is_engaged"] = (df["time_in_video"] >= threshold).astype(int)
    print(f"[data] Label distribution (threshold={threshold}s):")
    print(df["is_engaged"].value_counts().to_string())

    # --- Feature Engineering ---
    # Numeric features from XITE
    numeric_cols = []
    for col in ["session_order", "video_order", "time_in_video"]:
        if col in df.columns:
            # time_in_video is the target signal, don't use as feature
            if col != "time_in_video":
                numeric_cols.append(col)

    # Categorical features - encode genre/subgenres
    cat_cols = []
    if "genre" in df.columns:
        le_genre = LabelEncoder()
        df["genre_encoded"] = le_genre.fit_transform(df["genre"].fillna("unknown").astype(str))
        cat_cols.append("genre_encoded")

    if "subgenres" in df.columns:
        le_sub = LabelEncoder()
        df["subgenre_encoded"] = le_sub.fit_transform(df["subgenres"].fillna("unknown").astype(str))
        cat_cols.append("subgenre_encoded")

    # Release year as numeric
    for col in ["release_year", "release year"]:
        if col in df.columns:
            df["release_year_clean"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            numeric_cols.append("release_year_clean")
            break

    # Context segment if available
    if "context_segment" in df.columns:
        numeric_cols.append("context_segment")

    feature_cols = numeric_cols + cat_cols
    print(f"[data] Using features: {feature_cols}")

    df["_features"] = True  # marker
    return df, feature_cols


def split_by_session(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
    """
    Split data by session_id to prevent data leakage.
    All events in the same session stay in the same split.
    """
    print("[split] Splitting by session_id to prevent data leakage...")
    unique_sessions = df["session_id"].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_sessions)

    split_idx = int(len(unique_sessions) * train_ratio)
    train_sessions = set(unique_sessions[:split_idx])

    train_mask = df["session_id"].isin(train_sessions)
    train_df = df[train_mask]
    val_df = df[~train_mask]

    print(f"[split] Train sessions: {split_idx}, Val sessions: {len(unique_sessions) - split_idx}")
    print(f"[split] Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    return train_df, val_df


def train_logistic_regression(X_train, y_train, X_val, y_val, cfg):
    """Train a Logistic Regression baseline model."""
    print("[train] Training Logistic Regression baseline...")
    model_params = cfg.get("model_params", {})
    model = LogisticRegression(
        C=model_params.get("C", 1.0),
        max_iter=model_params.get("max_iter", 1000),
        solver=model_params.get("solver", "lbfgs"),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    return model, y_pred


def train_lightgbm(X_train, y_train, X_val, y_val, cfg):
    """Train a LightGBM model."""
    print("[train] Training LightGBM...")
    model_params = cfg.get("model_params", {})
    # Set defaults
    model_params.setdefault("objective", "binary")
    model_params.setdefault("metric", "binary_logloss")
    model_params.setdefault("verbosity", -1)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    num_boost_round = cfg.get("num_boost_round", 200)

    model = lgb.train(
        model_params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    y_pred = model.predict(X_val)
    return model, y_pred


def main():
    # --- Parse config ---
    if len(sys.argv) < 2:
        print("Usage: python train_ranking.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = load_config(config_path)
    print(f"[config] Loaded: {config_path}")
    print(f"[config] model_type={cfg['model_type']}")

    # --- MLFlow setup ---
    mlflow.set_experiment("smartqueue-stage-b")

    with mlflow.start_run(log_system_metrics=True):
        # Log all config params
        flat_params = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat_params[f"{k}.{k2}"] = v2
            else:
                flat_params[k] = v
        mlflow.log_params(flat_params)

        # Log git commit if available
        git_sha = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()
        if git_sha:
            mlflow.set_tag("git_sha", git_sha)

        # Log environment info (GPU or CPU)
        gpu_info = os.popen("nvidia-smi 2>/dev/null || echo 'No GPU - CPU only'").read()
        mlflow.log_text(gpu_info, "environment-info.txt")

        # --- Load data ---
        df, feature_cols = load_and_prepare_data(cfg)

        # --- Split ---
        seed = cfg.get("random_seed", 42)
        train_ratio = cfg.get("train_ratio", 0.8)
        train_df, val_df = split_by_session(df, train_ratio, seed)

        X_train = train_df[feature_cols].values
        y_train = train_df["is_engaged"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["is_engaged"].values

        # --- Train ---
        start_time = time.time()

        model_type = cfg["model_type"]
        if model_type == "logistic_regression":
            model, y_pred = train_logistic_regression(X_train, y_train, X_val, y_val, cfg)
        elif model_type == "lightgbm":
            model, y_pred = train_lightgbm(X_train, y_train, X_val, y_val, cfg)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        training_time = time.time() - start_time

        # --- Evaluate ---
        val_auc = roc_auc_score(y_val, y_pred)
        val_logloss = log_loss(y_val, y_pred)

        print(f"\n{'='*50}")
        print(f"[result] Validation AUC:     {val_auc:.4f}")
        print(f"[result] Validation LogLoss: {val_logloss:.4f}")
        print(f"[result] Training time:      {training_time:.1f}s")
        print(f"{'='*50}\n")

        # --- Log metrics ---
        mlflow.log_metrics({
            "val_auc": val_auc,
            "val_logloss": val_logloss,
            "training_time_seconds": training_time,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_features": len(feature_cols),
        })

        # --- Log model ---
        if model_type == "logistic_regression":
            mlflow.sklearn.log_model(model, "model")
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, "model")

        print("[done] Run logged to MLFlow successfully!")


if __name__ == "__main__":
    main()
