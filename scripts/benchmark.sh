#!/usr/bin/env bash
# benchmark.sh — Throughput benchmark for vLLM endpoints
#
# Usage:
#   # Benchmark a single endpoint
#   scripts/benchmark.sh --endpoint http://spark2:8000
#
#   # Compare FP16 baseline vs quantized candidate
#   scripts/benchmark.sh --baseline http://spark2:8000 --candidate http://spark2:8001
#
# Options:
#   --endpoint URL        Single endpoint to benchmark
#   --baseline URL        FP16 baseline endpoint (for A/B comparison)
#   --candidate URL       Quantized candidate endpoint (for A/B comparison)
#   --model NAME          Model name as reported by /v1/models (auto-detected if omitted)
#   --num-prompts N       Number of prompts to send (default: 100)
#   --concurrency N       Concurrent requests / request rate (default: 10)
#   --input-len N         Random input length in tokens (default: 512)
#   --output-len N        Random output length in tokens (default: 256)
#
# Prerequisites:
#   pip install aiohttp transformers
#   git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
ENDPOINT=""
BASELINE=""
CANDIDATE=""
MODEL=""
NUM_PROMPTS=100
CONCURRENCY=10
INPUT_LEN=512
OUTPUT_LEN=256
BENCHMARK_SCRIPT="/tmp/vllm/benchmarks/benchmark_serving.py"

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)     ENDPOINT="$2";    shift 2 ;;
    --baseline)     BASELINE="$2";    shift 2 ;;
    --candidate)    CANDIDATE="$2";   shift 2 ;;
    --model)        MODEL="$2";       shift 2 ;;
    --num-prompts)  NUM_PROMPTS="$2"; shift 2 ;;
    --concurrency)  CONCURRENCY="$2"; shift 2 ;;
    --input-len)    INPUT_LEN="$2";   shift 2 ;;
    --output-len)   OUTPUT_LEN="$2";  shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ── Validation ───────────────────────────────────────────────────────
if [[ -z "$ENDPOINT" && -z "$BASELINE" ]]; then
  echo "Error: Provide --endpoint for single benchmark or --baseline/--candidate for A/B comparison." >&2
  exit 1
fi

if [[ -n "$BASELINE" && -z "$CANDIDATE" ]]; then
  echo "Error: --baseline requires --candidate for A/B comparison." >&2
  exit 1
fi

# ── Ensure benchmark script exists ───────────────────────────────────
if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
  echo "Downloading vLLM benchmark scripts..."
  git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm 2>/dev/null || true
fi

if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
  echo "Error: Could not find $BENCHMARK_SCRIPT" >&2
  echo "Clone vLLM manually: git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm" >&2
  exit 1
fi

# ── Auto-detect model name from /v1/models ───────────────────────────
detect_model() {
  local url="$1"
  if command -v curl &>/dev/null && command -v python3 &>/dev/null; then
    curl -sf "${url}/v1/models" 2>/dev/null \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo ""
  fi
}

# ── Run benchmark against a single endpoint ──────────────────────────
run_benchmark() {
  local url="$1"
  local label="$2"
  local model_name="$3"

  if [[ -z "$model_name" ]]; then
    model_name=$(detect_model "$url")
    if [[ -z "$model_name" ]]; then
      echo "Error: Could not auto-detect model name from ${url}/v1/models." >&2
      echo "Specify --model explicitly." >&2
      exit 1
    fi
  fi

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  Benchmark: $label"
  echo "  Endpoint:  $url"
  echo "  Model:     $model_name"
  echo "  Prompts:   $NUM_PROMPTS  |  Concurrency: $CONCURRENCY"
  echo "  Input:     $INPUT_LEN tokens  |  Output: $OUTPUT_LEN tokens"
  echo "════════════════════════════════════════════════════════════════"
  echo ""

  python3 "$BENCHMARK_SCRIPT" \
    --backend openai-chat \
    --base-url "$url" \
    --endpoint /v1/chat/completions \
    --model "$model_name" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$CONCURRENCY"
}

# ── Main ─────────────────────────────────────────────────────────────
if [[ -n "$ENDPOINT" ]]; then
  # Single endpoint benchmark
  run_benchmark "$ENDPOINT" "Single Endpoint" "$MODEL"
else
  # A/B comparison
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║              A/B Benchmark: Baseline vs Candidate          ║"
  echo "╚══════════════════════════════════════════════════════════════╝"

  run_benchmark "$BASELINE" "Baseline (FP16)" "$MODEL"
  echo ""
  echo "────────────────────────────────────────────────────────────────"
  echo ""
  run_benchmark "$CANDIDATE" "Candidate (Quantized)" "$MODEL"

  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo "  Compare the metrics above side-by-side."
  echo "  Key indicators:"
  echo "    - TTFT (Time to First Token): lower is better"
  echo "    - TPOT (Time Per Output Token): lower is better"
  echo "    - Throughput (tokens/s): higher is better"
  echo ""
  echo "  For quality comparison, run:"
  echo "    lm_eval --model local-completions \\"
  echo "      --model_args 'model=<name>,base_url=<url>/v1' \\"
  echo "      --tasks hellaswag,mmlu,gsm8k --num_fewshot 5"
  echo "════════════════════════════════════════════════════════════════"
fi
