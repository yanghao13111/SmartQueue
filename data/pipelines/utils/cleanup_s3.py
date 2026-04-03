"""
Delete .keep placeholder files from Chameleon object storage.

Usage:
    python utils/cleanup_s3.py
"""

from s3 import BUCKET, list_objects, delete_objects

keys = [obj["Key"] for obj in list_objects() if obj["Key"].endswith(".keep")]

if not keys:
    print("No .keep files found.")
else:
    delete_objects(keys)
    for key in keys:
        print(f"  Deleted: {key}")
    print(f"\nDone. Removed {len(keys)} file(s).")
