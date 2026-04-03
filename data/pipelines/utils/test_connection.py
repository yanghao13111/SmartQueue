"""
Quick test to verify S3 connection to Chameleon object storage.
Run this first before pipeline to confirm credentials work.

Steps:
  1. Connect and list bucket contents
  2. Upload a .keep placeholder to each prefix (raw/, processed/, feedback/, retrain/)
  3. List bucket again to confirm

Usage:
    python utils/test_connection.py
"""

from s3 import BUCKET, get_client, list_objects, upload_file
import tempfile
from pathlib import Path

s3 = get_client()

# 1. Connect + list
try:
    contents = list_objects()
    print(f"[ok] Connected to bucket '{BUCKET}'")
    print(f"[ok] Objects in bucket: {len(contents)}")
    for obj in contents:
        print(f"  - {obj['Key']}  ({obj['Size']} bytes)")
except Exception as e:
    print(f"[error] {e}")
    raise SystemExit(1)

# 2. Upload .keep placeholders to each prefix
PREFIXES = ["raw/", "processed/", "feedback/", "retrain/"]
print("\nUploading .keep placeholders ...")
with tempfile.NamedTemporaryFile() as tmp:
    tmp_path = Path(tmp.name)
    for prefix in PREFIXES:
        key = f"{prefix}.keep"
        upload_file(tmp_path, key)
        print(f"  Uploaded: {key}")

# 3. List again to confirm
print("\nListing bucket after upload ...")
for obj in list_objects():
    print(f"  - {obj['Key']}  ({obj['Size']} bytes)")
