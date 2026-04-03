"""
Online Feature Computation

Computes user features in real-time from a session's listened events.
This mirrors the batch computation in feature_engineering.py but operates
on-demand for a single session, enabling real-time inference.

In production, this would be called before each /queue request:
  1. Take the events the user has listened to so far (first half of session)
  2. Compute user features from those events
  3. Combine with candidate songs → POST to /queue

Usage (demo):
    python feature_service.py [--data-dir /path/to/processed] [--session-id <id>]
"""

import json
import random
import argparse
from pathlib import Path

import pandas as pd

SKIP_THRESHOLD = 30
DEFAULT_DATA   = Path(__file__).resolve().parents[2] / "processed"


# ── core function ─────────────────────────────────────────────────────────────

def compute_user_features(events: list[dict]) -> dict:
    """
    Compute user features from a list of session events.

    Args:
        events: list of dicts with keys time_in_video (float), genre_encoded (int)

    Returns:
        dict with user_skip_rate, user_favorite_genre_encoded, user_watch_time_avg
    """
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


# ── demo ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default=str(DEFAULT_DATA))
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    prod_path = Path(args.data_dir) / "production.parquet"
    print(f"Loading {prod_path} ...")
    df = pd.read_parquet(prod_path)

    # Pick session
    session_id = args.session_id or random.choice(df["session_id"].unique())
    session    = df[df["session_id"] == session_id].reset_index(drop=True)
    half       = max(1, len(session) // 2)
    first_half = session.iloc[:half]
    second_half = session.iloc[half:]

    print(f"\nSession:    {session_id}")
    print(f"Total rows: {len(session)}  |  Using first {half} events for user profile")

    # Build events from first half
    events = [
        {
            "time_in_video": float(row["user_watch_time_avg"]),  # proxy since time_in_video not stored
            "genre_encoded": int(row["genre_encoded"]),
        }
        for _, row in first_half.iterrows()
    ]

    # Compute user features
    user_features = compute_user_features(events)

    print("\n── Computed User Features ──")
    print(json.dumps(user_features, indent=2))

    # Build sample /queue input
    candidates = [
        {
            "video_id":         str(row["video_id"]),
            "release_year":     int(row["release_year"]),
            "context_segment":  int(row["context_segment"]),
            "genre_encoded":    int(row["genre_encoded"]),
            "subgenre_encoded": int(row["subgenre_encoded"]),
        }
        for _, row in second_half.head(10).iterrows()
    ]
    queue_input = {
        "session_id":      session_id,
        "user_features":   user_features,
        "candidate_songs": candidates,
    }

    print("\n── Sample /queue Input ──")
    print(json.dumps(queue_input, indent=2))


if __name__ == "__main__":
    main()
