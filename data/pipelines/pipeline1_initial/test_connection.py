"""
Quick test to verify S3 connection to Chameleon object storage.
Run this first before pipeline to confirm credentials work.

Usage:
    python test_connection.py
"""

import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("S3_ENDPOINT", "https://chi.tacc.chameleoncloud.org:7480")
ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")
BUCKET = os.getenv("S3_BUCKET", "ObjStore_proj13")

s3 = boto3.client(
    's3',
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

try:
    resp = s3.list_objects_v2(Bucket=BUCKET)
    contents = resp.get('Contents', [])
    print(f"[ok] Connected to bucket '{BUCKET}'")
    print(f"[ok] Objects in bucket: {len(contents)}")
    for obj in contents:
        print(f"  - {obj['Key']}  ({obj['Size']} bytes)")
except ClientError as e:
    print(f"[error] {e}")
