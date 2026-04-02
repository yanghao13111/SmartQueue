# SmartQueue — Data Design Document

## 1. Overview

SmartQueue uses a two-phase data architecture:

1. **Offline Training Phase** — XITE session data (real + synthetic) is ingested into Chameleon object storage, transformed into feature sets, and used to train a LightGBM engagement ranking model.
2. **Production Simulation Phase** — A held-out production split from XITE emulates real users hitting the serving endpoint. Feedback (predicted vs. actual engagement) is logged per session and periodically folded back into training via a daily retrain pipeline to close the feedback loop.

All data lives in a single MinIO bucket (`smartqueue-data`) on Chameleon, organized by prefix:

```
smartqueue-data/
├── raw/                      # XITE source data (one-shot)
├── processed/                # Initial train/val/test/production splits
├── feedback/                 # Per-session feedback from production simulation
└── retrain/v{date}/          # Daily versioned retrain datasets
```

---

## 2. Data Repositories

### 2.1 Raw Data Storage

**Location:** `s3://smartqueue-data/raw/`

**Purpose:** Immutable landing zone for the external XITE source data. Only contains data ingested from outside — synthetic data is generated downstream in the feature pipeline.

| File | Description |
|------|-------------|
| `xite_msd.parquet` | Main XITE dataset — 31M events across 1M sessions |
| `metadata.json` | Ingestion timestamp, source URL, row counts |

**Schema — `xite_msd.parquet`:**

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | string | Unique session identifier |
| `video_id` | string | Unique music video identifier |
| `title` | string | Song title |
| `main_artist` | string | Primary artist |
| `secondary_artists` | string | Semicolon-delimited featured artists |
| `genre` | string | Primary genre (20 categories) |
| `subgenres` | string | Semicolon-delimited subgenres |
| `release_year` | int | Year of music video release |
| `video_isrc` | string | International Standard Recording Code |
| `time_in_video` | float | Seconds watched (engagement signal) |
| `mbid` | string | MusicBrainz recording ID |
| `mb_conf` | string | MusicBrainz match confidence (low/mid/high) |
| `session_order` | int | Session start-time rank (lower = earlier) |
| `video_order` | int | 0-indexed position of video within session |
| `context_segment` | int | Viewing context grouping (playlist, search, etc.) |

**Written by:** Ingestion pipeline (one-shot)
**Versioned by:** `metadata.json` with ingestion timestamp and row counts

---

### 2.2 Processed Feature Store

**Location:** `s3://smartqueue-data/processed/`

**Purpose:** Initial train/val/test/production splits with pre-computed features, generated once before the first training run. Synthetic rows are generated here and mixed into train/val splits.

| File | Description |
|------|-------------|
| `train.parquet` | Training examples (real + synthetic rows mixed) |
| `val.parquet` | Validation examples (real + synthetic rows mixed) |
| `test.parquet` | Held-out test examples (real data only) |
| `production.parquet` | Production simulation split — real data only, emulates live users |
| `metadata.json` | Split sizes, label distribution, timestamp |

**Schema — all split parquet files:**

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | string | Session identifier |
| `video_id` | string | Video identifier |
| `is_engaged` | int (0/1) | Label: 1 if time_in_video ≥ 30s |
| `genre_encoded` | int | LabelEncoded genre |
| `subgenre_encoded` | int | LabelEncoded primary subgenre |
| `release_year` | int | Year of release |
| `context_segment` | int | Viewing context ID |
| `user_skip_rate` | float | Fraction of skips across all events in the session |
| `user_favorite_genre_encoded` | int | Mode genre (LabelEncoded) across all events in the session |
| `user_watch_time_avg` | float | Mean watch time (seconds) across all events in the session |

**Note on user features:** For training and evaluation, user features are computed from all events in the session and attached to every row. For production simulation, user features are computed from only the first half of the session to mirror real serving behavior.

**Written by:** Feature pipeline (Pipeline 1, one-shot)
**Versioned by:** `metadata.json` with timestamp and split sizes

---

### 2.3 Production Feedback Store

**Location:** `s3://smartqueue-data/feedback/`

**Purpose:** Per-session feedback logs from production simulation. The data generator accumulates feedback in memory during a session, then writes one file per session when the session ends. These files are the input to the daily retrain pipeline.

| File | Description |
|------|-------------|
| `{YYYYMMDD}_{session_id}.jsonl` | One JSON record per ranked song in this session |

**Schema — each JSONL record:**

```json
{
  "session_id": "766d672f-...",
  "video_id": "901a48d1-...",
  "rank_position": 1,
  "predicted_engagement_prob": 0.85,
  "actual_time_in_video": 22.0,
  "actual_is_engaged": 0,
  "timestamp": "2026-04-02T10:00:00Z"
}
```

**Write pattern:** The generator holds feedback records in memory during simulation. At session end, all records are written to MinIO in a single PUT as `{YYYYMMDD}_{session_id}.jsonl`. This avoids concurrent write conflicts when multiple sessions run in parallel, and allows the retrain pipeline to filter by date prefix.

**Written by:** Data generator (production simulation script)
**Versioned by:** Each session gets its own immutable file; no overwriting

---

### 2.4 Retrain Dataset Store

**Location:** `s3://smartqueue-data/retrain/v{YYYYMMDD}/`

**Purpose:** Daily versioned training datasets produced by the retrain pipeline. Each version is a full regeneration — not an append — combining original XITE data with accumulated feedback.

| File | Description |
|------|-------------|
| `train.parquet` | Updated training set (XITE + feedback rows) |
| `metadata.json` | Date, feedback sessions included, row counts, label distribution |

**How feedback rows are incorporated:**
1. Read all `feedback/{YYYYMMDD}_{session_id}.jsonl` files matching today's date prefix
2. Join on `video_id` against `raw/xite_msd.parquet` to recover video features (genre, release_year, etc.)
3. Run same feature engineering as Pipeline 1
4. Merge with original `processed/train.parquet`
5. Write to `retrain/v{date}/train.parquet`

**Written by:** Retrain pipeline (Pipeline 2, runs daily)
**Versioned by:** Date-prefixed folder `v{YYYYMMDD}` — previous versions are never overwritten

---

## 3. Data Flow

```mermaid
flowchart TD
    A[XITE Million Sessions Dataset\nexternal source] -->|Pipeline 1: ingestion| B[(Raw Storage\nraw/)]
    B -->|Pipeline 1: feature engineering\n+ synthetic generation| C[(Processed Feature Store\nprocessed/)]
    C -->|train split| D[Training Pipeline]
    C -->|production split| E[Data Generator\nproduction simulation]
    E -->|POST /queue| G[FastAPI Serving\n/queue endpoint]
    G -->|ranked_songs| E
    E -->|write at session end| H[(Feedback Store\nfeedback/)]
    H -->|Pipeline 2: retrain| I[(Retrain Dataset\nretrain/v{date}/)]
    I -->|updated train.parquet| D

    style B fill:#2d4a7a,color:#fff
    style C fill:#2d4a7a,color:#fff
    style H fill:#2d4a7a,color:#fff
    style I fill:#2d4a7a,color:#fff
```

---

## 4. Synthetic Data Generation

XITE (~2.5 GB) is below the 5 GB threshold, so synthetic sessions are generated to expand the dataset.

**Strategy: Cross-Session User Profile Mixing**

For each synthetic session:
1. Sample a "user" session (session A) to derive `user_skip_rate` and `user_favorite_genre_encoded`
2. Sample a "content" session (session B) to source candidate videos and their `time_in_video` ground truth
3. Combine: user profile from A + video events from B
4. Add Gaussian noise to `time_in_video` (σ = 5s) to avoid exact duplication
5. Re-derive label: `is_engaged = (noisy_time_in_video >= 30)`

**Quality control:**
- Synthetic label distribution (skip rate) must stay within ±5% of original distribution
- Sessions A and B must come from the same split (no cross-split leakage)

Synthetic data is generated inside Pipeline 1 and mixed directly into `train.parquet` and `val.parquet`. `test.parquet` and `production.parquet` contain real data only.

---

## 5. Versioning Strategy

| Repository | Versioning Mechanism |
|------------|---------------------|
| Raw Storage | Immutable; `metadata.json` with ingestion timestamp and row counts |
| Processed Feature Store | Immutable after generation; `metadata.json` with timestamp and split sizes |
| Feedback Store | One immutable file per session; never overwritten |
| Retrain Dataset Store | Date-prefixed folder `v{YYYYMMDD}`; previous versions retained |

---

## 6. Data Pipeline Summary

| Pipeline | Trigger | Input | Output | Location |
|----------|---------|-------|--------|----------|
| Pipeline 1: Ingestion + Feature | One-shot | XITE download | `raw/` + `processed/` | `data/pipelines/pipeline1_initial/` |
| Pipeline 2: Daily Retrain | Daily (or manual) | `raw/xite_msd.parquet` + `feedback/*.jsonl` | `retrain/v{date}/` | `data/pipelines/pipeline2_retrain/` |
| Data Generator | Manual simulation run | `processed/production.parquet` | `feedback/{YYYYMMDD}_{session_id}.jsonl` | `data/pipelines/generator/` |
