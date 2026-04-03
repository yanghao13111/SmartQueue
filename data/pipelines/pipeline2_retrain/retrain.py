"""
Pipeline 2 - Batch Retrain

Reads feedback JSONL files, joins with production.parquet to recover video
and user features, and produces a versioned retrain dataset.

User profile update strategy:
  - Start from the pre-computed user features stored in production.parquet
  - Merge in the actual events from feedback (time_in_video proxy via is_engaged)
  - Re-compute user features over the combined history
  - This means the same session accumulates a richer profile across generator loops

Output:
  data/retrain/v{YYYYMMDD}/
    train.parquet   — original train + new feedback rows
    metadata.json   — feedback count, label distribution, timestamp

Usage:
    python retrain.py [--data-dir /path/to/SmartQueue/data]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
import s3

LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"

SKIP_THRESHOLD = 30
SCRIPT_DIR     = Path(__file__).resolve().parent
DEFAULT_DATA   = Path(os.getenv("DATA_DIR", "/app/data"))


# ── user feature computation (same logic as feature_service.py) ───────────────

def compute_user_features(events: list[dict]) -> dict:
    times  = [e["time_in_video"] for e in events]
    genres = [e["genre_encoded"]  for e in events]

    skip_rate = round(sum(1 for t in times if t < SKIP_THRESHOLD) / len(times), 4)
    watch_avg = round(sum(times) / len(times), 2)

    genre_counts = {}
    for g in genres:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    fav_genre = max(genre_counts, key=genre_counts.get)

    return {
        "user_skip_rate":              skip_rate,
        "user_favorite_genre_encoded": fav_genre,
        "user_watch_time_avg":         watch_avg,
    }


# ── load feedback ─────────────────────────────────────────────────────────────

def load_feedback(feedback_dir: Path, date_str: str) -> pd.DataFrame:
    files = list(feedback_dir.glob(f"{date_str}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No feedback files found for date {date_str} in {feedback_dir}")

    print(f"  Found {len(files):,} feedback files for {date_str}")
    records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} feedback records from {df['session_id'].nunique():,} sessions")
    return df


# ── build retrain rows ────────────────────────────────────────────────────────

def build_retrain_rows(feedback_df: pd.DataFrame, production_df: pd.DataFrame) -> pd.DataFrame:
    # Join feedback with production to get video features
    prod_video = production_df[["video_id", "genre_encoded", "subgenre_encoded",
                                "release_year", "context_segment"]].drop_duplicates("video_id")
    merged = feedback_df.merge(prod_video, on="video_id", how="inner")

    if len(merged) < len(feedback_df):
        dropped = len(feedback_df) - len(merged)
        print(f"  Warning: {dropped:,} feedback rows dropped (video_id not in production)")

    # Get pre-computed user profile per session from production
    prod_profile = production_df.groupby("session_id")[
        ["user_skip_rate", "user_favorite_genre_encoded", "user_watch_time_avg"]
    ].first()

    # Join real time_in_video from production.parquet (preserved for production split)
    prod_time = production_df[["session_id", "video_id", "time_in_video"]]
    merged = merged.merge(prod_profile, on="session_id", how="left")
    merged = merged.merge(prod_time, on=["session_id", "video_id"], how="left")
    merged["time_in_video"] = merged["time_in_video"].fillna(10.0)

    # Re-compute user features per session (precomputed + feedback events merged)
    rows = []
    for session_id, group in merged.groupby("session_id"):
        # Combine pre-stored profile as synthetic events + actual feedback events
        pre = prod_profile.loc[session_id] if session_id in prod_profile.index else None

        events = []
        if pre is not None:
            # Add pre-computed profile as a synthetic "history" event
            events.append({
                "time_in_video": float(pre["user_watch_time_avg"]),
                "genre_encoded": int(pre["user_favorite_genre_encoded"]),
            })
        for _, row in group.iterrows():
            events.append({
                "time_in_video": float(row["time_in_video"]),
                "genre_encoded": int(row["genre_encoded"]),
            })

        uf = compute_user_features(events)

        for _, row in group.iterrows():
            rows.append({
                "session_id":                  session_id,
                "video_id":                    row["video_id"],
                "is_engaged":                  int(row["actual_is_engaged"]),
                "genre_encoded":               int(row["genre_encoded"]),
                "subgenre_encoded":            int(row["subgenre_encoded"]),
                "release_year":                int(row["release_year"]),
                "context_segment":             int(row["context_segment"]),
                "user_skip_rate":              uf["user_skip_rate"],
                "user_favorite_genre_encoded": uf["user_favorite_genre_encoded"],
                "user_watch_time_avg":         uf["user_watch_time_avg"],
            })

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--date", default=None,
                        help="Date to process (YYYYMMDD). Defaults to today (UTC).")
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    feedback_dir  = data_dir / "feedback"
    processed_dir = data_dir / "processed"
    date_str      = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
    retrain_dir   = data_dir / "retrain" / f"v{date_str}"
    retrain_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    # 1. Load feedback
    print("\n[1/4] Loading feedback ...")
    t = time.perf_counter()
    feedback_df = load_feedback(feedback_dir, date_str)
    print(f"[1/4] Done in {time.perf_counter()-t:.1f}s")

    # 2. Load production.parquet
    print("\n[2/4] Loading production.parquet ...")
    t = time.perf_counter()
    production_df = pd.read_parquet(processed_dir / "production.parquet")
    print(f"  {len(production_df):,} rows  ({time.perf_counter()-t:.1f}s)")

    # 3. Build feedback rows with updated user features
    print("\n[3/4] Building retrain rows ...")
    t = time.perf_counter()
    new_rows = build_retrain_rows(feedback_df, production_df)
    print(f"  {len(new_rows):,} new rows built  ({time.perf_counter()-t:.1f}s)")

    # 4. Merge with original train.parquet
    print("\n[4/4] Merging with original train.parquet ...")
    t = time.perf_counter()
    train_df  = pd.read_parquet(processed_dir / "train.parquet")
    retrain   = pd.concat([train_df, new_rows], ignore_index=True)
    out_path  = retrain_dir / "train.parquet"
    retrain.to_parquet(out_path, index=False)
    print(f"  Original: {len(train_df):,}  +  New: {len(new_rows):,}  =  Total: {len(retrain):,}")
    print(f"  Saved → {out_path}  ({time.perf_counter()-t:.1f}s)")

    # Metadata
    metadata = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "version":            f"v{date_str}",
        "feedback_files":     len(list(feedback_dir.glob(f"{date_str}_*.jsonl"))),
        "feedback_sessions":  int(feedback_df["session_id"].nunique()),
        "feedback_rows":      int(len(new_rows)),
        "original_train_rows": int(len(train_df)),
        "total_rows":         int(len(retrain)),
        "engaged_rate":       round(float(retrain["is_engaged"].mean()), 4),
    }
    with open(retrain_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.perf_counter() - total_start
    print(f"\n{'='*50}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Retrain data → {retrain_dir}")
    print(f"{'='*50}")

    # Upload retrain output to S3
    if not LOCAL_MODE:
        print("\n[5/4] Uploading to S3 ...")
        t = time.perf_counter()
        s3_prefix = f"retrain/v{date_str}"
        n = s3.upload_dir(retrain_dir, s3_prefix)
        print(f"  {n} file(s) → s3://{s3.BUCKET}/{s3_prefix}/  ({time.perf_counter()-t:.1f}s)")
    else:
        print("\nLOCAL_MODE=true — skipping S3 upload.")


if __name__ == "__main__":
    main()
