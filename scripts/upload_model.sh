#!/usr/bin/env bash
# upload_model.sh — Upload a cloned model directory to MinIO.
#
# Prerequisites:
#   1. MinIO is running (see minio/docker-compose.yml)
#   2. Install the MinIO client (mc):
#        curl -O https://dl.min.io/client/mc/release/linux-arm64/mc   # DGX Spark (ARM)
#        curl -O https://dl.min.io/client/mc/release/linux-amd64/mc   # x86_64
#        chmod +x mc && sudo mv mc /usr/local/bin/
#
# Usage:
#   # First-time setup — configure the MinIO alias:
#   mc alias set myminio http://10.x.x.20:9000 minioadmin <password>
#
#   # Upload a model:
#   ./scripts/upload_model.sh /models/Qwen3-Embedding-8B
#   ./scripts/upload_model.sh /models/Qwen3-8B
#   ./scripts/upload_model.sh /models/Qwen3-72B-AWQ
#
# This creates the following structure in MinIO:
#   s3://models/Qwen3-Embedding-8B/config.json
#   s3://models/Qwen3-Embedding-8B/tokenizer.json
#   s3://models/Qwen3-Embedding-8B/model-00001-of-00004.safetensors
#   ...

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 /path/to/model-directory}"
MINIO_ALIAS="${MINIO_ALIAS:-myminio}"
BUCKET="${MINIO_BUCKET:-models}"
MODEL_NAME="$(basename "$MODEL_DIR")"

# Validate
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "Error: $MODEL_DIR does not look like a model directory (no config.json found)."
    exit 1
fi

# Check for LFS pointer files (common gotcha)
for f in "$MODEL_DIR"/*.safetensors; do
    [[ -f "$f" ]] || continue
    size=$(stat --printf="%s" "$f")
    if (( size < 1000 )); then
        echo "Error: $f is only $size bytes — likely an LFS pointer, not actual weights."
        echo "Run:  cd $MODEL_DIR && git lfs pull"
        exit 1
    fi
done

# Create bucket if it doesn't exist
mc mb --ignore-existing "${MINIO_ALIAS}/${BUCKET}"

# Upload with progress — mirror preserves directory structure.
# --exclude '.git*' avoids uploading git metadata.
echo "Uploading $MODEL_DIR → s3://${BUCKET}/${MODEL_NAME}/"
mc mirror --overwrite --exclude '.git*' "$MODEL_DIR" "${MINIO_ALIAS}/${BUCKET}/${MODEL_NAME}"

echo ""
echo "Done. Model available at: s3://${BUCKET}/${MODEL_NAME}"
echo ""
echo "To list uploaded files:"
echo "  mc ls ${MINIO_ALIAS}/${BUCKET}/${MODEL_NAME}/"
echo ""
echo "To use with vLLM (direct S3 pull):"
echo "  --model s3://${BUCKET}/${MODEL_NAME}"
echo ""
echo "To sync to a node (hybrid approach):"
echo "  mc mirror ${MINIO_ALIAS}/${BUCKET}/${MODEL_NAME} /models/${MODEL_NAME}"
