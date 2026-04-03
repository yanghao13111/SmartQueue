"""
Pipeline 1 - Step 3: Upload to S3

Uploads raw/ and processed/ to Chameleon object storage.
Skipped when LOCAL_MODE=true.

Usage:
    python upload_data.py [--data-dir /path/to/SmartQueue/data]
"""

import argparse
import sys
import time
from pathlib import Path

# Resolve utils/ relative to this file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import s3

LOCAL_MODE = __import__("os").getenv("LOCAL_MODE", "false").lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                        help="Path to SmartQueue/data. Defaults to two levels above this script.")
    args = parser.parse_args()

    if LOCAL_MODE:
        print("LOCAL_MODE=true — skipping S3 upload.")
        return

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parents[2]
    total_start = time.perf_counter()

    print("\n[3/3] Uploading to S3 ...")

    for folder in ("raw", "processed"):
        local_dir = data_dir / folder
        if not local_dir.exists():
            raise FileNotFoundError(f"{folder}/ not found at {local_dir}")
        print(f"  {folder}/")
        t = time.perf_counter()
        n = s3.upload_dir(local_dir, folder)
        print(f"  {n} file(s)  ({time.perf_counter()-t:.1f}s)")

    print(f"\n  Total upload time: {time.perf_counter()-total_start:.1f}s")
    print(f"  Bucket: s3://{s3.BUCKET}/")


if __name__ == "__main__":
    main()
