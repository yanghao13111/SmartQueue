"""
Data Generator — SmartQueue

Simulates production users by replaying the production split against the
/queue serving endpoint. For each session:
  - User features come from the pre-computed first-half profile in production.parquet
  - 10 randomly sampled songs from the second half are sent as candidates
  - The ranked response (or mock if no endpoint configured) is used to build feedback
  - Feedback is written locally and uploaded to S3 at session end

Runs indefinitely (loops through all sessions), cycling through again on completion.

Usage:
    python generator.py [options]

Environment variables:
    QUEUE_ENDPOINT      URL of /queue endpoint (default: mock mode)
    S3_ENDPOINT         Chameleon Object Storage endpoint
    S3_ACCESS_KEY       S3 access key
    S3_SECRET_KEY       S3 secret key
    S3_BUCKET           S3 bucket name (default: ObjStore_proj13)
    LOCAL_MODE          If 'true', skip S3 download/upload (default: false)
    CONCURRENCY         Number of concurrent sessions (default: 50)
    CANDIDATES_PER_REQ  Songs to sample per request (default: 10)
    FEEDBACK_DELAY      Seconds between processing each ranked song (default: 0.5)
"""

import os
import json
import random
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

QUEUE_ENDPOINT     = os.getenv("QUEUE_ENDPOINT", "")
S3_ENDPOINT        = os.getenv("S3_ENDPOINT", "")
S3_ACCESS_KEY      = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY      = os.getenv("S3_SECRET_KEY", "")
S3_BUCKET          = os.getenv("S3_BUCKET", "ObjStore_proj13")
LOCAL_MODE         = os.getenv("LOCAL_MODE", "true").lower() == "true"
CONCURRENCY        = int(os.getenv("CONCURRENCY", "10"))
CANDIDATES_PER_REQ = int(os.getenv("CANDIDATES_PER_REQ", "10"))
FEEDBACK_DELAY     = float(os.getenv("FEEDBACK_DELAY", "1.0"))

SCRIPT_DIR    = Path(__file__).resolve().parent
# Docker default: /app/data
# Local override: set DATA_DIR env var to /path/to/SmartQueue/data
DEFAULT_DATA  = Path(os.getenv("DATA_DIR", "/app/data"))
FEEDBACK_DIR  = DEFAULT_DATA / "feedback"
PROCESSED_DIR = DEFAULT_DATA / "processed"
PROD_PARQUET  = PROCESSED_DIR / "production.parquet"
S3_PROD_KEY   = "processed/production.parquet"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def download_production_parquet():
    if LOCAL_MODE:
        if not PROD_PARQUET.exists():
            raise FileNotFoundError(f"LOCAL_MODE=true but {PROD_PARQUET} not found")
        log.info(f"LOCAL_MODE: using {PROD_PARQUET}")
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if PROD_PARQUET.exists():
        log.info(f"production.parquet already exists locally, skipping download")
        return
    log.info(f"Downloading s3://{S3_BUCKET}/{S3_PROD_KEY} ...")
    s3 = get_s3_client()
    s3.download_file(S3_BUCKET, S3_PROD_KEY, str(PROD_PARQUET))
    log.info(f"Downloaded → {PROD_PARQUET}")


def upload_feedback(local_path: Path, s3_key: str):
    if LOCAL_MODE:
        return
    s3 = get_s3_client()
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    log.info(f"S3 uploaded → {s3_key}")


# ── queue call ────────────────────────────────────────────────────────────────

def call_queue(session_id: str, user_features: dict, candidates: list[dict]) -> list[dict]:
    """Call /queue endpoint or fall back to mock ranking."""
    payload = {
        "session_id": session_id,
        "user_features": user_features,
        "candidate_songs": candidates,
    }

    if not QUEUE_ENDPOINT:
        # mock: random engagement probabilities
        ranked = [
            {
                "video_id": c["video_id"],
                "engagement_probability": round(random.random(), 4),
                "rank": i + 1,
            }
            for i, c in enumerate(random.sample(candidates, len(candidates)))
        ]
        return ranked

    try:
        resp = requests.post(QUEUE_ENDPOINT, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["ranked_songs"]
    except Exception as e:
        log.warning(f"Queue endpoint failed ({e}), falling back to mock")
        return [
            {
                "video_id": c["video_id"],
                "engagement_probability": round(random.random(), 4),
                "rank": i + 1,
            }
            for i, c in enumerate(random.sample(candidates, len(candidates)))
        ]


# ── session processing ────────────────────────────────────────────────────────

async def process_session(
    session_id: str,
    session_df: pd.DataFrame,
    run_id: int,
    loop_num: int,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        loop = asyncio.get_event_loop()

        # User features (pre-computed from first half, same for all rows in session)
        row0 = session_df.iloc[0]
        user_features = {
            "user_skip_rate":              float(row0["user_skip_rate"]),
            "user_favorite_genre_encoded": int(row0["user_favorite_genre_encoded"]),
            "user_watch_time_avg":         float(row0["user_watch_time_avg"]),
        }

        # Sample up to CANDIDATES_PER_REQ songs from this session
        sample = session_df.sample(min(CANDIDATES_PER_REQ, len(session_df)), random_state=run_id)
        candidates = [
            {
                "video_id":        str(row["video_id"]),
                "release_year":    int(row["release_year"]),
                "context_segment": int(row["context_segment"]),
                "genre_encoded":   int(row["genre_encoded"]),
                "subgenre_encoded": int(row["subgenre_encoded"]),
            }
            for _, row in sample.iterrows()
        ]

        # Call /queue (run in thread to avoid blocking event loop)
        ranked = await loop.run_in_executor(
            None, call_queue, session_id, user_features, candidates
        )

        # Build ground truth lookup
        gt = {str(row["video_id"]): int(row["is_engaged"]) for _, row in sample.iterrows()}

        # Process each ranked song with delay
        records = []
        for song in ranked:
            vid = song["video_id"]
            engaged = gt.get(vid, 0)
            prob = song["engagement_probability"]
            rec = {
                "session_id":                session_id,
                "video_id":                  vid,
                "rank_position":             song["rank"],
                "predicted_engagement_prob": prob,
                "actual_is_engaged":         engaged,
                "timestamp":                 datetime.now(timezone.utc).isoformat(),
            }
            records.append(rec)
            log.info(
                f"[loop {loop_num}] {session_id[:8]}... "
                f"rank={song['rank']:2d}  prob={prob:.3f}  engaged={engaged}"
            )
            await asyncio.sleep(FEEDBACK_DELAY)

        # Write JSONL locally
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = f"{date_str}_{session_id}_{loop_num}_{run_id}.jsonl"
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        local_path = FEEDBACK_DIR / filename
        with open(local_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Upload to S3
        s3_key = f"feedback/{filename}"
        await loop.run_in_executor(None, upload_feedback, local_path, s3_key)

        log.info(f"[loop {loop_num}] session {session_id[:8]}... done → {len(records)} records saved to {filename}")


# ── main loop ─────────────────────────────────────────────────────────────────

async def run(sessions_df: pd.DataFrame, session_ids: list):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    loop_num = 0
    run_counter = 0

    # Pre-index by session_id for fast lookup
    session_map = {sid: grp for sid, grp in sessions_df.groupby("session_id")}

    log.info(f"Starting generator: {len(session_ids):,} sessions, concurrency={CONCURRENCY}, "
             f"candidates={CANDIDATES_PER_REQ}, delay={FEEDBACK_DELAY}s")
    log.info(f"Queue endpoint: {QUEUE_ENDPOINT or 'MOCK MODE'}")
    log.info(f"S3 upload: {'disabled (LOCAL_MODE)' if LOCAL_MODE else S3_BUCKET}")

    while True:
        loop_num += 1
        log.info(f"── Loop {loop_num} starting ({len(session_ids):,} sessions) ──")

        shuffled = random.sample(session_ids, len(session_ids))

        # Process in batches to avoid creating 50k tasks at once
        batch_size = CONCURRENCY * 10
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            tasks = []
            for session_id in batch:
                run_counter += 1
                tasks.append(
                    process_session(session_id, session_map[session_id], run_counter, loop_num, semaphore)
                )
            await asyncio.gather(*tasks)

        log.info(f"── Loop {loop_num} complete ──")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=0,
                        help="Number of sessions to run per loop (0 = all)")
    args = parser.parse_args()

    download_production_parquet()

    log.info(f"Loading {PROD_PARQUET} ...")
    df = pd.read_parquet(PROD_PARQUET)
    session_ids = df["session_id"].unique().tolist()

    if args.sessions > 0:
        session_ids = session_ids[:args.sessions]
        log.info(f"Using {len(session_ids):,} sessions (--sessions {args.sessions})")

    asyncio.run(run(df[df["session_id"].isin(session_ids)], session_ids))


if __name__ == "__main__":
    main()
