"""
SmartQueue Stage B - Optuna Hyperparameter Tuning for LightGBM
Uses TPE sampler (Bayesian) to efficiently explore the hyperparameter space.
Each trial is logged as a nested MLflow run under one parent "optuna" run.

Usage:
    python tune_ranking.py stage_b_lgbm_v2.yaml
    python tune_ranking.py stage_b_lgbm_v2.yaml --n-trials 50 --study-name my-study
"""

import io
import os
import sys
import time
import yaml
import argparse
import optuna
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# Reuse data loading / splitting from the main training script
sys.path.insert(0, os.path.dirname(__file__))
from train_ranking import load_and_prepare_data, split_by_session


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val, base_cfg: dict) -> float:
    """
    One Optuna trial: sample hyperparams → train LightGBM → return val AUC.
    Each trial is logged as a *nested* MLflow run so it appears under the
    parent run in the MLflow UI.
    """
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        # ── Search space ──────────────────────────────────────────────────────
        # Log-uniform for learning rate — small values often matter more
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # Regularisation — log-uniform so tiny and large values are both reachable
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    num_boost_round = trial.suggest_int("num_boost_round", 100, 500)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    start = time.time()
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.log_evaluation(period=-1),           # suppress per-round output
            lgb.early_stopping(stopping_rounds=30, verbose=False),
        ],
    )
    elapsed = time.time() - start

    y_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, y_pred)
    val_logloss = log_loss(y_val, y_pred)

    # Log this trial as a nested MLflow run
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        mlflow.set_tag("optuna_trial_number", trial.number)
        # Log all sampled params
        mlflow.log_param("num_boost_round", num_boost_round)
        for k, v in params.items():
            if k not in ("objective", "metric", "verbosity"):
                mlflow.log_param(k, v)
        mlflow.log_metrics({
            "val_auc": val_auc,
            "val_logloss": val_logloss,
            "training_time_seconds": elapsed,
            "best_iteration": model.best_iteration,
        })

    print(
        f"[trial {trial.number:3d}] AUC={val_auc:.4f}  logloss={val_logloss:.4f}"
        f"  time={elapsed:.1f}s  best_iter={model.best_iteration}"
    )
    return val_auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for SmartQueue Stage B LightGBM")
    parser.add_argument("config", help="Base YAML config (data settings will be reused)")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--study-name", default="stage-b-lgbm-optuna", help="Optuna study name")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["model_type"] = "lightgbm"  # always LightGBM for tuning
    seed = cfg.get("random_seed", 42)

    print(f"[config] Base config: {args.config}")
    print(f"[optuna] n_trials={args.n_trials}, study={args.study_name}")

    mlflow.set_experiment("smartqueue-stage-b")

    # One parent run wraps all trials so they're grouped in the MLflow UI
    with mlflow.start_run(run_name=f"optuna_{args.study_name}"):
        mlflow.set_tags({
            "tuning_method": "optuna_tpe",
            "study_name": args.study_name,
        })
        mlflow.log_params({
            "n_trials": args.n_trials,
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
            "base_config": os.path.basename(args.config),
        })

        # Log environment info
        gpu_info = os.popen("nvidia-smi 2>/dev/null || echo 'No GPU - CPU only'").read()
        mlflow.log_text(gpu_info, "environment-info.txt")

        # ── Load & split data once (reused across all trials) ────────────────
        df, feature_cols = load_and_prepare_data(cfg)
        train_df, val_df = split_by_session(df, cfg.get("train_ratio", 0.8), seed)

        X_train = train_df[feature_cols].values
        y_train = train_df["is_engaged"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["is_engaged"].values

        mlflow.log_params({
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_features": len(feature_cols),
        })

        # ── Create Optuna study ───────────────────────────────────────────────
        # TPE (Tree-structured Parzen Estimator): Bayesian-style, far better than
        # grid or random search at this trial budget.
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        # MedianPruner: after n_startup_trials warm-up, prune trials whose
        # intermediate val metric falls below the median of completed trials.
        # (We don't use step-level pruning here since lgb early_stopping handles it.)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",   # maximise val_auc
            sampler=sampler,
            pruner=pruner,
        )

        # ── Run optimisation ──────────────────────────────────────────────────
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, cfg),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # ── Report & log best result ──────────────────────────────────────────
        best = study.best_trial
        print(f"\n{'='*60}")
        print(f"[optuna] Best trial : #{best.number}")
        print(f"[optuna] Best val AUC: {best.value:.4f}")
        print(f"[optuna] Best params :")
        for k, v in best.params.items():
            print(f"           {k}: {v}")
        print(f"{'='*60}\n")

        mlflow.log_metric("best_val_auc", best.value)
        mlflow.log_param("best_trial_number", best.number)
        for k, v in best.params.items():
            mlflow.log_param(f"best.{k}", v)

        # Save best params as a ready-to-use YAML config artifact
        best_cfg = {
            "model_type": "lightgbm",
            "data_path": cfg["data_path"],
            "skip_threshold_seconds": cfg.get("skip_threshold_seconds", 30),
            "max_samples": cfg.get("max_samples", 200000),
            "random_seed": seed,
            "train_ratio": cfg.get("train_ratio", 0.8),
            "num_boost_round": best.params.pop("num_boost_round", 300),
            "model_params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                **{k: v for k, v in best.params.items()},
            },
        }
        buf = io.StringIO()
        yaml.dump(best_cfg, buf, default_flow_style=False, sort_keys=False)
        mlflow.log_text(buf.getvalue(), "best_config.yaml")

        print("[done] Optuna tuning complete. All trials logged to MLflow.")


if __name__ == "__main__":
    main()
