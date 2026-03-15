#!/usr/bin/env bash
# sync_model.sh — Download a model from MinIO to local disk.
#
# This is the "hybrid approach": models are stored centrally in MinIO,
# but vLLM serves from local disk to avoid S3 API overhead during inference
# and to decouple MinIO availability from serving.
#
# Usage:
#   # Sync a specific model:
#   ./scripts/sync_model.sh Qwen3-Embedding-8B
#
#   # Sync all models in the bucket:
#   ./scripts/sync_model.sh --all
#
# Prerequisites:
#   mc alias set myminio http://10.x.x.20:9000 minioadmin <password>

set -euo pipefail

MINIO_ALIAS="${MINIO_ALIAS:-myminio}"
BUCKET="${MINIO_BUCKET:-models}"
LOCAL_BASE="${LOCAL_MODEL_DIR:-/models}"

sync_model() {
    local model_name="$1"
    local dest="${LOCAL_BASE}/${model_name}"

    echo "Syncing s3://${BUCKET}/${model_name} → ${dest}/ ..."
    mkdir -p "$dest"
    mc mirror --overwrite "${MINIO_ALIAS}/${BUCKET}/${model_name}" "$dest"

    # Verify weights aren't truncated
    for f in "$dest"/*.safetensors; do
        [[ -f "$f" ]] || continue
        size=$(stat --printf="%s" "$f")
        if (( size < 1000 )); then
            echo "Warning: $f is only $size bytes — may be corrupted."
        fi
    done

    echo "Done: ${dest}"
}

if [[ "${1:-}" == "--all" ]]; then
    echo "Syncing all models from s3://${BUCKET}/ ..."
    while IFS= read -r line; do
        # mc ls output format: "[date] [size] model_name/"
        model_name=$(echo "$line" | awk '{print $NF}' | tr -d '/')
        [[ -n "$model_name" ]] && sync_model "$model_name"
    done < <(mc ls "${MINIO_ALIAS}/${BUCKET}/")
elif [[ -n "${1:-}" ]]; then
    sync_model "$1"
else
    echo "Usage: $0 <model-name>    # sync one model"
    echo "       $0 --all           # sync all models"
    echo ""
    echo "Available models in s3://${BUCKET}/:"
    mc ls "${MINIO_ALIAS}/${BUCKET}/"
    exit 1
fi
