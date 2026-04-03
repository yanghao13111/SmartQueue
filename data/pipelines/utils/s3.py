"""
Shared S3 utilities for Chameleon object storage.

Environment variables (loaded from .env or passed in):
    S3_ENDPOINT   — e.g. https://chi.tacc.chameleoncloud.org:7480
    S3_ACCESS_KEY
    S3_SECRET_KEY
    S3_BUCKET     — default: ObjStore_proj13
"""

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

ENDPOINT   = os.getenv("S3_ENDPOINT",   "https://chi.tacc.chameleoncloud.org:7480")
ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")
BUCKET     = os.getenv("S3_BUCKET",     "ObjStore_proj13")


def get_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


def upload_file(local_path: Path, s3_key: str, bucket: str = BUCKET):
    """Upload a single file to S3."""
    get_client().upload_file(str(local_path), bucket, s3_key)


def upload_dir(local_dir: Path, s3_prefix: str, bucket: str = BUCKET):
    """Upload all files in a directory recursively, preserving relative structure."""
    local_dir = Path(local_dir)
    s3 = get_client()
    uploaded = 0
    for f in sorted(local_dir.rglob("*")):
        if not f.is_file():
            continue
        key = f"{s3_prefix.rstrip('/')}/{f.relative_to(local_dir)}"
        s3.upload_file(str(f), bucket, key)
        print(f"  Uploaded: {key}")
        uploaded += 1
    return uploaded


def list_objects(prefix: str = "", bucket: str = BUCKET) -> list[dict]:
    """Return list of objects (Key, Size) under a prefix."""
    resp = get_client().list_objects_v2(Bucket=bucket, Prefix=prefix)
    return resp.get("Contents", [])


def delete_objects(keys: list[str], bucket: str = BUCKET):
    """Delete a list of S3 keys."""
    if not keys:
        return
    get_client().delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": k} for k in keys]},
    )
