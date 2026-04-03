"""
Pipeline 1 - Step 2: Feature Engineering

Flow:
  1. Load raw xite_msd.parquet + encode categoricals
  2. Split session IDs → save temporary raw split parquets
  3. For each split: compute user features + label + synthetic (train/val only)
  4. Save final processed parquets + metadata.json
  5. Delete temporary splits directory

Split ratios: train=80%, val=10%, test=5%, production=5%
User features:
  - train/val/test: computed from ALL events in session
  - production:     computed from FIRST HALF of session events only
Synthetic data: train and val only (cross-session user profile mixing)

Usage:
    python feature_engineering.py [--output-dir /path/to/SmartQueue/data]
"""

import json
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

SKIP_THRESHOLD  = 30
TRAIN_RATIO     = 0.80
VAL_RATIO       = 0.10
TEST_RATIO      = 0.05
PROD_RATIO      = 0.05
RANDOM_SEED     = 42
SYNTH_NOISE_STD = 5.0
SYNTH_EXTRA_PCT = 1.0

NEEDED_COLS = [
    "session_id", "video_id",
    "genre", "subgenres",
    "release_year", "context_segment",
    "time_in_video", "video_order",
]

FINAL_COLS = [
    "session_id", "video_id", "is_engaged",
    "genre_encoded", "subgenre_encoded",
    "release_year", "context_segment",
    "user_skip_rate", "user_favorite_genre_encoded", "user_watch_time_avg",
]


# ── step 1: load & encode ─────────────────────────────────────────────────────

def load_and_encode(raw_dir: Path):
    path = raw_dir / "xite_msd.parquet"
    print(f"\n[1/4] Loading {path} ...")
    t = time.perf_counter()
    df = pd.read_parquet(path, columns=NEEDED_COLS)
    print(f"      {len(df):,} rows, {df['session_id'].nunique():,} sessions  ({time.perf_counter()-t:.1f}s)")

    print("[1/4] Encoding genre + subgenre ...")
    t = time.perf_counter()
    le_genre = LabelEncoder()
    df["genre_encoded"] = le_genre.fit_transform(
        df["genre"].fillna("unknown").astype(str)
    )
    df["subgenre_primary"] = (
        df["subgenres"].fillna("unknown").astype(str)
        .str.split(";").str[0].str.strip()
    )
    le_sub = LabelEncoder()
    df["subgenre_encoded"] = le_sub.fit_transform(df["subgenre_primary"])
    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    ).fillna(0).astype(int)
    print(f"      Genres: {len(le_genre.classes_)}, Subgenres: {len(le_sub.classes_)}  ({time.perf_counter()-t:.1f}s)")
    return df, le_genre, le_sub


# ── step 2: split sessions → raw split parquets ───────────────────────────────

def split_and_save_raw(df: pd.DataFrame, splits_dir: Path, seed: int = RANDOM_SEED):
    splits_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n[2/4] Splitting {df['session_id'].nunique():,} sessions → {splits_dir} ...")

    sessions = np.array(df["session_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = int(n * TEST_RATIO)

    split_ids = {
        "train":      sessions[:n_train],
        "val":        sessions[n_train:n_train + n_val],
        "test":       sessions[n_train + n_val:n_train + n_val + n_test],
        "production": sessions[n_train + n_val + n_test:],
    }

    for name, ids in split_ids.items():
        t = time.perf_counter()
        out = splits_dir / f"{name}_raw.parquet"
        if out.exists():
            print(f"  {out.name} already exists, skipping")
            continue
        subset = df[df["session_id"].isin(ids)]
        subset.to_parquet(out, index=False)
        print(f"  {out.name}  {len(subset):,} rows ({len(ids):,} sessions)  {time.perf_counter()-t:.1f}s")

    print(f"[2/4] Split done in {time.perf_counter()-t0:.1f}s")
    return split_ids


# ── step 3a: user features (vectorized, full session) ────────────────────────

def user_features_full(subset: pd.DataFrame) -> pd.DataFrame:
    g = subset.groupby("session_id")
    return pd.DataFrame({
        "user_skip_rate": g["time_in_video"].apply(
            lambda x: (x < SKIP_THRESHOLD).mean()
        ).round(4),
        "user_favorite_genre_encoded": g["genre_encoded"].agg(
            lambda x: int(x.mode().iloc[0])
        ),
        "user_watch_time_avg": g["time_in_video"].mean().round(2),
    }).reset_index()


# ── step 3b: user features (vectorized, first half) ──────────────────────────

def user_features_first_half(subset: pd.DataFrame) -> pd.DataFrame:
    s = subset.copy()
    s["_rank"] = s.groupby("session_id")["video_order"].rank(method="first")
    s["_len"]  = s.groupby("session_id")["video_order"].transform("count")
    fh = s[s["_rank"] <= (s["_len"] / 2)]
    g  = fh.groupby("session_id")
    return pd.DataFrame({
        "user_skip_rate": g["time_in_video"].apply(
            lambda x: (x < SKIP_THRESHOLD).mean()
        ).round(4),
        "user_favorite_genre_encoded": g["genre_encoded"].agg(
            lambda x: int(x.mode().iloc[0])
        ),
        "user_watch_time_avg": g["time_in_video"].mean().round(2),
    }).reset_index()


# ── step 3c: synthetic data (bulk vectorized) ────────────────────────────────

def generate_synthetic(feat_df: pd.DataFrame, n_extra: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    sessions = feat_df["session_id"].unique()

    # Pre-compute one profile row per session (columns are already in feat_df)
    profiles = feat_df.groupby("session_id")[
        ["user_skip_rate", "user_favorite_genre_encoded", "user_watch_time_avg"]
    ].first()

    avg_len  = len(feat_df) / len(sessions)
    n_pairs  = int(n_extra / avg_len * 1.4) + 500   # slight overshot, trim later

    print(f"  Sampling {n_pairs:,} pairs (avg {avg_len:.1f} events/session) ...")
    sess_a = rng.choice(sessions, size=n_pairs)
    sess_b = rng.choice(sessions, size=n_pairs)
    valid  = sess_a != sess_b
    sess_a, sess_b = sess_a[valid], sess_b[valid]

    # ── single bulk filter: get all sess_b events at once ────────────────────
    event_cols = ["session_id", "video_id", "genre_encoded", "subgenre_encoded",
                  "release_year", "context_segment", "time_in_video"]
    print(f"  Bulk-fetching events for {len(set(sess_b)):,} unique sess_b ...")
    b_events = (
        feat_df.loc[feat_df["session_id"].isin(set(sess_b)), event_cols]
        .rename(columns={"session_id": "session_b"})
    )

    # ── build pair mapping table ──────────────────────────────────────────────
    pair_map = pd.DataFrame({
        "session_b":      sess_b,
        "session_a":      sess_a,
        "new_session_id": np.arange(len(sess_a)),   # int key, rename to string after trim
    })

    # ── one merge expands all events for all pairs ────────────────────────────
    print(f"  Merging pairs × events ...")
    synth = b_events.merge(pair_map, on="session_b", how="inner")

    # ── vectorized noise + label ──────────────────────────────────────────────
    noise = rng.normal(0, SYNTH_NOISE_STD, size=len(synth))
    synth["time_in_video"] = (synth["time_in_video"].values + noise).clip(min=0)
    synth["is_engaged"]    = (synth["time_in_video"] >= SKIP_THRESHOLD).astype(int)

    # ── attach sess_a user profile ────────────────────────────────────────────
    synth = synth.merge(profiles, left_on="session_a", right_index=True, how="left")

    # ── trim to exact n_extra, then assign readable session IDs ──────────────
    synth = synth.head(n_extra).copy()
    synth["session_id"] = "synthetic_" + synth["new_session_id"].astype(str)

    orig_e  = feat_df["is_engaged"].mean()
    synth_e = synth["is_engaged"].mean()
    diff    = abs(orig_e - synth_e)
    print(f"  label dist — original: {orig_e:.3f}, synthetic: {synth_e:.3f}, diff: {diff:.3f}")
    if diff > 0.05:
        print(f"  WARNING: diff {diff:.3f} > 0.05")

    return synth[FINAL_COLS]


# ── step 3+4: process each split ─────────────────────────────────────────────

def process_split(name: str, splits_dir: Path, processed_dir: Path) -> dict:
    raw_path = splits_dir / f"{name}_raw.parquet"
    out_path = processed_dir / f"{name}.parquet"
    t0 = time.perf_counter()

    if out_path.exists():
        print(f"  {out_path.name} already exists, skipping")
        df = pd.read_parquet(out_path)
        return {
            "rows": len(df),
            "engaged_rate": round(float(df["is_engaged"].mean()), 4),
            "skip_rate":    round(float(1 - df["is_engaged"].mean()), 4),
        }

    t = time.perf_counter()
    print(f"  Loading {raw_path.name} ...")
    subset = pd.read_parquet(raw_path)
    n_sessions = subset["session_id"].nunique()
    print(f"  Loaded {len(subset):,} rows  ({time.perf_counter()-t:.1f}s)")

    use_first_half = (name == "production")
    t = time.perf_counter()
    print(f"  Computing user features (first_half={use_first_half}) ...")
    uf = user_features_first_half(subset) if use_first_half else user_features_full(subset)
    print(f"  User features done  ({time.perf_counter()-t:.1f}s)")

    subset = subset.merge(uf, on="session_id", how="left")
    subset["is_engaged"] = (subset["time_in_video"] >= SKIP_THRESHOLD).astype(int)
    print(f"  {len(subset):,} rows  engaged_rate={subset['is_engaged'].mean():.3f}")

    if name in ("train", "val"):
        n_extra  = int(len(subset) * SYNTH_EXTRA_PCT)
        print(f"  Generating {n_extra:,} synthetic rows ...")
        t = time.perf_counter()
        synth_df = generate_synthetic(subset[FINAL_COLS + ["time_in_video"]], n_extra)
        print(f"  Synthetic done  ({time.perf_counter()-t:.1f}s)")
        subset   = pd.concat([subset[FINAL_COLS], synth_df], ignore_index=True)
        print(f"  Total after synthetic: {len(subset):,}")
    else:
        subset = subset[FINAL_COLS]

    t = time.perf_counter()
    subset.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}  ({time.perf_counter()-t:.1f}s)")
    print(f"  [{name}] total: {time.perf_counter()-t0:.1f}s")

    return {
        "sessions":     int(n_sessions),
        "rows":         int(len(subset)),
        "engaged_rate": round(float(subset["is_engaged"].mean()), 4),
        "skip_rate":    round(float(1 - subset["is_engaged"].mean()), 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    default_output = str(Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--output-dir", default=default_output)
    args = parser.parse_args()

    base_dir      = Path(args.output_dir)
    raw_dir       = base_dir / "raw"
    splits_dir    = base_dir / "splits"
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    total_start   = time.perf_counter()

    # 1. Load + encode
    t = time.perf_counter()
    df, le_genre, le_sub = load_and_encode(raw_dir)
    print(f"[1/4] Done in {time.perf_counter()-t:.1f}s")

    # 2. Split → save raw splits
    t = time.perf_counter()
    split_and_save_raw(df, splits_dir)
    del df
    print(f"[2/4] Done in {time.perf_counter()-t:.1f}s")

    # 3+4. Process each split
    print("\n[3/4] Processing splits ...")
    results = {}
    for name in ["train", "val", "test", "production"]:
        print(f"\n  ── {name} ──")
        results[name] = process_split(name, splits_dir, processed_dir)

    # 4. Metadata
    print("\n[4/4] Writing metadata ...")
    t = time.perf_counter()
    metadata = {
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "skip_threshold_seconds": SKIP_THRESHOLD,
        "split_ratios":           {"train": TRAIN_RATIO, "val": VAL_RATIO,
                                   "test": TEST_RATIO, "production": PROD_RATIO},
        "user_features_note":     "train/val/test use all session events; production uses first half only",
        "splits":                 results,
        "genre_classes":          len(le_genre.classes_),
        "subgenre_classes":       len(le_sub.classes_),
    }
    meta_path = processed_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[4/4] Done in {time.perf_counter()-t:.1f}s")

    # 5. Clean up temporary splits
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
        print(f"Cleaned up {splits_dir}")

    elapsed = time.perf_counter() - total_start
    print(f"\n{'='*50}")
    print(f"Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"Processed data → {processed_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
