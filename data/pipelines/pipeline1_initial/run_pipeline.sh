#!/bin/bash
# Pipeline 1: Ingestion + Feature Engineering
# Runs both steps sequentially.
#
# Usage:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --output-dir /tmp/smartqueue

set -e

SCRIPT_DIR_TMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${OUTPUT_DIR:-"$(cd "$SCRIPT_DIR_TMP/../.." && pwd)"}

# Parse optional --output-dir argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$SCRIPT_DIR_TMP"

echo "============================================"
echo " SmartQueue Pipeline 1: Ingestion + Feature"
echo " Output dir: $OUTPUT_DIR"
echo "============================================"

PYTHON=${PYTHON:-python3}

echo ""
echo "--- Installing dependencies ---"
$PYTHON -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "--- Step 1: Ingestion ---"
INGEST_ARGS="--output-dir $OUTPUT_DIR"
if [ -n "$SOURCE" ]; then
    INGEST_ARGS="$INGEST_ARGS --source $SOURCE"
fi
$PYTHON "$SCRIPT_DIR/ingest.py" $INGEST_ARGS

echo ""
echo "--- Step 2: Feature Engineering ---"
$PYTHON "$SCRIPT_DIR/feature_engineering.py" --output-dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo " Pipeline 1 complete."
echo " Output: $OUTPUT_DIR/processed/"
echo "============================================"
