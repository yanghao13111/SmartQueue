"""
Pipeline 1 - Step 1: Ingestion
Copies local XITE parquet to output directory and writes metadata.json.

Usage:
    python ingest.py [--output-dir /tmp/smartqueue]
    python ingest.py --output-dir /tmp/smartqueue --source /path/to/xite_msd.parquet
"""

import os
import json
import argparse
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone

import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()

XITE_URL = "https://millionsessionsdataset.xite.com/xite_msd.zip"

# Default: look for local file relative to repo root
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "XITE-Million-Sessions-Dataset" / "xite_msd.parquet"


def copy_parquet(source: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "xite_msd.parquet"

    if dest.exists():
        print(f"[ingest] {dest} already exists, skipping copy")
        return dest

    print(f"[ingest] Copying {source} → {dest} ...")
    shutil.copy2(source, dest)
    print(f"[ingest] Copy complete ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def write_metadata(raw_dir: Path, row_count: int):
    metadata = {
        "source_url": XITE_URL,
        "source_note": "Data ingested from local copy of XITE Million Sessions Dataset",
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "parquet_file": "xite_msd.parquet",
    }
    meta_path = raw_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[ingest] Metadata written: {meta_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    default_output = str(Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--output-dir", default=default_output, help="Base output directory")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to local xite_msd.parquet")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source parquet not found: {source}\nPlease download XITE from {XITE_URL}")

    raw_dir = Path(args.output_dir) / "raw"
    total_start = time.perf_counter()

    # Step 1: Copy parquet
    t = time.perf_counter()
    dest = copy_parquet(source, raw_dir)
    print(f"[ingest] Step 1 done in {time.perf_counter() - t:.1f}s")

    # Step 2: Count rows
    t = time.perf_counter()
    print("[ingest] Reading row count...")
    pf = pq.ParquetFile(dest)
    row_count = pf.metadata.num_rows
    print(f"[ingest] Row count: {row_count:,}  ({time.perf_counter() - t:.1f}s)")

    # Step 3: Write metadata
    t = time.perf_counter()
    write_metadata(raw_dir, row_count)
    print(f"[ingest] Step 3 done in {time.perf_counter() - t:.1f}s")

    print(f"\n[ingest] Done. Total: {time.perf_counter() - total_start:.1f}s  Raw data at: {raw_dir}")
    return dest


if __name__ == "__main__":
    main()
