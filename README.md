# Self-Hosting LLMs on NVIDIA DGX Spark

> **TL;DR:** This is a practical, end-to-end guide for deploying embedding and chat completion models on NVIDIA DGX Spark systems using vLLM, with LiteLLM as a unified API gateway. It covers local testing on a single node, production deployment across multiple nodes, and client integration — all designed for air-gapped or restricted corporate networks where models must be cloned from ModelScope rather than pulled from Hugging Face.

## Repository Structure

```
.gitignore                # Ignores .env files, model weights, logs

spark1/                   # Deploy on DGX Spark #1 (embedding node)
  docker-compose.yml
  .env.example

spark2/                   # Deploy on DGX Spark #2 (chat node)
  docker-compose.yml
  .env.example

gateway/                  # Deploy on the gateway host (no GPU needed)
  docker-compose.yml      # LiteLLM + nginx
  litellm-config.yaml     # Model routing, health checks, failover
  nginx.conf              # Reverse proxy (HTTP + optional HTTPS)
  .env.example

minio/                    # Optional: centralized S3-compatible model store
  docker-compose.yml
  .env.example

monitoring/               # Optional: Prometheus + Grafana
  docker-compose.yml
  prometheus.yml

scripts/
  smoke_test.py           # Validate endpoints (direct vLLM or through LiteLLM)
  health_check.sh         # Cron-based liveness check with alerting
  upload_model.sh         # Upload a cloned model directory to MinIO
  sync_model.sh           # Pull a model from MinIO to local disk
```

## Table of Contents
- [Repository Structure](#repository-structure)
- [Context and Constraints](#context-and-constraints)
- [Hardware Overview: DGX Spark](#hardware-overview-dgx-spark)
- [Prerequisites](#prerequisites)
- [Acquiring Models in a Restricted Network](#acquiring-models-in-a-restricted-network)
- [Centralized Model Storage with MinIO](#centralized-model-storage-with-minio-optional)
- [Part 1: Local Testing](#part-1-local-testing)
- [Part 2: Production Deployment](#part-2-production-deployment)
- [Part 3: Client Integration](#part-3-client-integration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Quantization: Running Larger Models](#quantization-running-larger-models)
- [Zero-Downtime Model Updates](#zero-downtime-model-updates)
- [Redundancy and Failure Handling](#redundancy-and-failure-handling)
- [Troubleshooting](#troubleshooting)
- [Key Takeaways](#key-takeaways)
- [References](#references)

## Context and Constraints

This guide assumes the following environment:

| Constraint | Detail |
|---|---|
| **Hardware** | 2x NVIDIA DGX Spark systems |
| **Network** | Corporate/air-gapped — no access to Hugging Face Hub or ModelScope APIs |
| **Model source** | Git clone from ModelScope (e.g., `git clone https://www.modelscope.ai/...`) |
| **Serving framework** | vLLM (OpenAI-compatible API) |
| **API gateway** | LiteLLM (unified routing, API key management, usage tracking) |
| **Target models** | Qwen3-Embedding-8B (embeddings), Qwen3-8B or larger (chat completion) |

### Why This Setup?

```mermaid
graph LR
    A[Users] -->|OpenAI SDK| N["nginx<br/>llm.example.com"]
    N --> B[LiteLLM Gateway]
    B -->|"model: qwen3-embed"| C["DGX Spark #1<br/>vLLM — Embedding"]
    B -->|"model: qwen3-chat"| D["DGX Spark #2<br/>vLLM — Chat"]

    style N fill:#9b59b6,stroke:#8e44ad,color:#fff
    style B fill:#f39c12,stroke:#c87f0a,color:#fff
    style C fill:#3498db,stroke:#2980b9,color:#fff
    style D fill:#2ecc71,stroke:#27ae60,color:#fff
```

- **vLLM** provides high-throughput serving with PagedAttention, continuous batching, and an OpenAI-compatible API out of the box.
- **LiteLLM** sits in front as a unified gateway, giving users a single endpoint with clean model names, API key management, usage tracking, and rate limiting.
- **nginx** sits between users and LiteLLM, handling TLS termination and letting clients connect on standard ports (80/443) without specifying a port number.
- **DGX Spark** offers 128 GB of unified memory per node — enough to run 8B models in FP16 comfortably, or quantized models up to ~70B.

## Hardware Overview: DGX Spark

Each DGX Spark system provides:

| Spec | Value |
|---|---|
| **Chip** | NVIDIA GB10 Grace Blackwell Superchip |
| **Architecture** | ARM-based (sm_121) |
| **CPU** | 20 cores (10x Cortex-X925 + 10x Cortex-A725) |
| **Memory** | 128 GB unified LPDDR5x (shared CPU/GPU) |
| **Memory bandwidth** | 273 GB/s |
| **AI compute** | Up to 1 PFLOP (FP4) |
| **Storage** | 4 TB SSD |
| **Power** | 240W |

**Key architectural note:** The DGX Spark uses a **unified memory architecture** — CPU and GPU share the same 128 GB address space. There is no separate VRAM. This eliminates CPU-to-GPU transfer overhead but means all system processes share the memory pool.

### What Fits on a DGX Spark?

| Model | Precision | Memory Required | Fits? |
|---|---|---|---|
| Qwen3-Embedding-8B | FP16 | ~16 GB | Yes — plenty of headroom |
| Qwen3-8B | FP16 | ~16 GB | Yes |
| Qwen3-32B | FP16 | ~64 GB | Yes, with limited KV cache |
| Qwen3-30B-A3B (MoE) | FP16 | ~60 GB | Yes — only 3B active params |
| Qwen3-72B | INT4 (GPTQ/AWQ) | ~36 GB | Yes, quantized |
| Qwen3-72B | FP16 | ~144 GB | No — exceeds 128 GB |

## Prerequisites

### 1. Git LFS

Model weights are stored as Git LFS objects. Git LFS must be installed and configured **before** cloning:

```bash
# Install Git LFS
sudo apt-get install git-lfs   # Debian/Ubuntu
# or
sudo dnf install git-lfs       # RHEL/Fedora

# Initialize (one-time)
git lfs install
```

> **Air-gapped note:** If `git lfs` is unavailable or blocked, you can download `.safetensors` weight files individually from the ModelScope web interface and place them in the model directory manually. The directory structure must match what the model's `config.json` expects.

### 2. Docker and NVIDIA Container Toolkit

NVIDIA provides a pre-built vLLM Docker image optimized for DGX Spark's sm_121 architecture. Ensure Docker is configured for GPU access:

```bash
# Verify Docker and NVIDIA Container Toolkit are installed
docker --version
nvidia-smi

# Ensure your user can run Docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.8-base nvidia-smi
```

### 3. vLLM Compatibility on DGX Spark

**This is the most important prerequisite to understand.** The DGX Spark uses the sm_121 GPU architecture, which upstream vLLM does not fully support as of early 2026. You have two options:

| Option | Pros | Cons |
|---|---|---|
| **NVIDIA's pre-built vLLM container** | Works out of the box, tested for DGX Spark | May lag behind upstream vLLM features |
| **Build vLLM from source** | Latest features, full control | Requires patching Triton for sm_121, more complex setup |

The NVIDIA container is available from NGC. As of this writing, the latest version is `26.02-py3`:

```bash
# Set the version (check https://build.nvidia.com/spark/vllm for the latest)
export VLLM_VERSION=26.02-py3
docker pull nvcr.io/nvidia/vllm:${VLLM_VERSION}
```

**System requirements:** CUDA 13.0 toolkit, Python 3.12, Docker with NVIDIA Container Toolkit.

The `--enforce-eager` flag is **required** on DGX Spark — it disables CUDA graph optimizations that aren't compatible with sm_121. This results in approximately 20-30% slower inference compared to CUDA graph mode on supported architectures, but it is the only reliable option.

### NVIDIA-Quantized Models for DGX Spark

NVIDIA provides pre-quantized models optimized for DGX Spark on [build.nvidia.com](https://build.nvidia.com/spark/vllm). These use NVFP4 and FP8 formats tuned for Blackwell:

| Model | Quantization | Handle |
|---|---|---|
| Llama-3.1-8B-Instruct | FP8 / NVFP4 | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| Llama-3.3-70B-Instruct | NVFP4 | `nvidia/Llama-3.3-70B-Instruct-NVFP4` |
| Qwen3-8B | FP8 / NVFP4 | `nvidia/Qwen3-8B-FP8` |
| Nemotron-3-Super-120B-A12B | FP8 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` |
| Nemotron-3-Nano-30B-A3B | BF16 / FP8 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Phi-4-multimodal-instruct | FP8 / NVFP4 | `nvidia/Phi-4-multimodal-instruct-FP8` |

> These can be used as an alternative to ModelScope models if your network allows pulling from Hugging Face via the NVIDIA container's built-in download. If not, the ModelScope git clone approach described below still applies.

## Acquiring Models in a Restricted Network

### Clone from ModelScope

```bash
# Create a central model store
sudo mkdir -p /models
sudo chown $USER:$USER /models

# Clone embedding model
git clone https://www.modelscope.ai/Qwen/Qwen3-Embedding-8B.git /models/Qwen3-Embedding-8B

# Clone chat model
git clone https://www.modelscope.ai/Qwen/Qwen3-8B.git /models/Qwen3-8B
```

### Verify Model Files

After cloning, confirm the weight files are actual tensors (not LFS pointers):

```bash
# Check file sizes — safetensors files should be multiple GB, not 130 bytes
ls -lh /models/Qwen3-Embedding-8B/*.safetensors

# If files are tiny (~130 bytes), LFS didn't pull the actual weights
# Fix with:
cd /models/Qwen3-Embedding-8B && git lfs pull
```

### Qwen3-Embedding-8B Model Details

| Property | Value |
|---|---|
| **Parameters** | 8 billion |
| **Max context length** | 32,768 tokens |
| **Embedding dimensions** | 32 to 4,096 (configurable, Matryoshka) |
| **Default dimension** | 4,096 |
| **FP16 memory** | ~16 GB |

The model uses Matryoshka Representation Learning, meaning you can truncate embeddings to smaller dimensions (e.g., 512, 1024) while preserving most semantic quality — useful for reducing storage and search costs in vector databases.

---

## Centralized Model Storage with MinIO (Optional)

Instead of manually copying model weights to each DGX Spark node, you can run MinIO as a centralized S3-compatible model store. This simplifies model distribution and versioning — upload once, pull from any node.

### Why MinIO?

| Concern | Without MinIO | With MinIO |
|---|---|---|
| **Adding a model to a new node** | `scp` or `rsync` multi-GB files manually | `mc mirror myminio/models/Qwen3-8B /models/Qwen3-8B` |
| **Model updates** | Repeat the copy to every node | Upload once; nodes sync on restart |
| **Versioning / rollback** | Hope you kept the old directory | MinIO bucket versioning or named prefixes |
| **vLLM integration** | Local path only | Local path (recommended) or `s3://` URI directly |

### Setup

```bash
# Deploy MinIO (see minio/docker-compose.yml)
cd minio
cp .env.example .env && nano .env   # set MINIO_ROOT_PASSWORD
docker compose up -d

# Install the MinIO client (mc) — one-time
# ARM (DGX Spark):
curl -O https://dl.min.io/client/mc/release/linux-arm64/mc
# x86_64:
# curl -O https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc && sudo mv mc /usr/local/bin/

# Configure the alias
mc alias set myminio http://10.x.x.20:9000 minioadmin <your-password>
```

### Upload Models

After cloning from ModelScope, upload to MinIO:

```bash
# Upload (excludes .git metadata, verifies weights aren't LFS pointers)
./scripts/upload_model.sh /models/Qwen3-Embedding-8B
./scripts/upload_model.sh /models/Qwen3-8B
```

### Pull Models to a Node

**Recommended: sync to local disk, serve from disk** (hybrid approach). This decouples MinIO availability from inference — if MinIO goes down, running vLLM instances are unaffected:

```bash
# On each DGX Spark node
./scripts/sync_model.sh Qwen3-Embedding-8B    # → /models/Qwen3-Embedding-8B
./scripts/sync_model.sh Qwen3-8B              # → /models/Qwen3-8B
```

Then serve from `/models/` as usual (the default in `spark1/` and `spark2/` Docker Compose files).

**Alternative: vLLM pulls directly from S3**. Simpler but adds startup latency (vLLM downloads all weights before serving) and makes MinIO a runtime dependency:

```bash
# In docker-compose.yml, set these environment variables:
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=<your-password>
AWS_ENDPOINT_URL=http://10.x.x.20:9000

# And change the vllm serve command:
vllm serve s3://models/Qwen3-Embedding-8B --task embed ...
```

Both `spark1/docker-compose.yml` and `spark2/docker-compose.yml` include the S3/MinIO configuration as commented-out blocks.

---

## Part 1: Local Testing

Use this section to validate your setup on a single DGX Spark before deploying to production.

### 1.1 Pull the vLLM Container

```bash
# NVIDIA's DGX Spark optimized image (check https://build.nvidia.com/spark/vllm for latest tag)
export VLLM_VERSION=26.02-py3
docker pull nvcr.io/nvidia/vllm:${VLLM_VERSION}
```

> **Air-gapped image transfer:** If the DGX Spark can't reach a container registry, export the image on a machine that can and transfer it:
>
> ```bash
> # On a machine with registry access
> docker pull nvcr.io/nvidia/vllm:26.02-py3
> docker save nvcr.io/nvidia/vllm:26.02-py3 | gzip > vllm-26.02-py3.tar.gz
>
> # Copy to the DGX Spark (USB drive, scp, shared NFS, etc.)
> scp vllm-26.02-py3.tar.gz user@dgx-spark:/tmp/
>
> # On the DGX Spark
> gunzip -c /tmp/vllm-26.02-py3.tar.gz | docker load
> ```
>
> Do the same for `ghcr.io/berriai/litellm`, `nginx:alpine`, and any other images needed.

### 1.2 Test the Embedding Model

```bash
docker run --rm --gpus all \
  -v /models/Qwen3-Embedding-8B:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --task embed \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --enforce-eager \
    --dtype float16
```

Key flags explained:

| Flag | Purpose |
|---|---|
| `--task embed` | Tells vLLM to serve as an embedding model (pooling runner) |
| `--enforce-eager` | **Required on DGX Spark** — disables incompatible CUDA graphs |
| `--max-model-len 8192` | Limits context window to save memory; increase up to 32768 if needed |
| `--dtype float16` | Use FP16 precision; change to `bfloat16` if supported |

Verify it's running:

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Generate an embedding
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "input": "This is a test sentence for embedding."
  }'
```

Expected: a JSON response with an `embedding` array of 4,096 floats.

### 1.3 Test the Chat Model

Stop the embedding container first (or use a different port), then:

```bash
docker run --rm --gpus all \
  -v /models/Qwen3-8B:/model \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 8192 \
    --enforce-eager \
    --dtype float16
```

Verify:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [
      {"role": "user", "content": "Explain what vLLM is in two sentences."}
    ],
    "max_tokens": 128
  }'
```

### 1.4 Test Both on One Node (Optional)

An 8B embedding model (~16 GB) and an 8B chat model (~16 GB) can coexist on one DGX Spark (128 GB unified memory). This is useful for local testing but not recommended for production throughput:

```bash
# Terminal 1: Embedding model on port 8000
docker run --rm --gpus all \
  -v /models/Qwen3-Embedding-8B:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --task embed \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --enforce-eager --dtype float16 \
    --gpu-memory-utilization 0.4

# Terminal 2: Chat model on port 8001
docker run --rm --gpus all \
  -v /models/Qwen3-8B:/model \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 8192 \
    --enforce-eager --dtype float16 \
    --gpu-memory-utilization 0.4
```

The `--gpu-memory-utilization 0.4` flag limits each instance to 40% of available GPU memory, preventing OOM when running side by side.

### 1.5 Quick Smoke Test Script

`scripts/smoke_test.py` supports both direct vLLM testing (this section) and testing through LiteLLM (Part 2). In direct mode, the model name is the container path (`/model`) rather than the LiteLLM alias (`qwen3-embed` / `qwen3-chat`).

```bash
pip install openai

# Test directly against vLLM (embed on :8000, chat on :8001)
python scripts/smoke_test.py --mode direct

# Test through LiteLLM gateway (after Part 2 setup)
python scripts/smoke_test.py --mode gateway \
  --base-url http://llm.example.com/v1 \
  --api-key sk-your-team-key
```

---

## Part 2: Production Deployment

### 2.1 Architecture

```mermaid
graph TD
    Users["Users<br/>(OpenAI SDK, curl, apps)"]
    Nginx["nginx Reverse Proxy<br/>llm.example.com:80/443"]
    LiteLLM["LiteLLM Proxy<br/>localhost:4000"]
    Spark1["DGX Spark #1<br/>vLLM — Embedding<br/>10.x.x.10:8000"]
    Spark2["DGX Spark #2<br/>vLLM — Chat<br/>10.x.x.11:8000"]
    Prometheus["Prometheus"]
    Grafana["Grafana Dashboard"]

    Users -->|"POST /v1/embeddings<br/>POST /v1/chat/completions"| Nginx
    Nginx --> LiteLLM
    LiteLLM -->|"model: qwen3-embed"| Spark1
    LiteLLM -->|"model: qwen3-chat"| Spark2
    Spark1 -->|"/metrics"| Prometheus
    Spark2 -->|"/metrics"| Prometheus
    Prometheus --> Grafana

    style Nginx fill:#9b59b6,stroke:#8e44ad,color:#fff
    style LiteLLM fill:#f39c12,stroke:#c87f0a,color:#fff
    style Spark1 fill:#3498db,stroke:#2980b9,color:#fff
    style Spark2 fill:#2ecc71,stroke:#27ae60,color:#fff
    style Prometheus fill:#e74c3c,stroke:#c0392b,color:#fff
```

**One model per node** — this maximizes available memory for KV cache and concurrent requests.

### 2.2 Deploy vLLM on DGX Spark #1 (Embedding)

Use `spark1/` from this repo. The Docker Compose file includes healthchecks, GPU resource reservations, configurable env vars for context length / memory utilization / concurrency, and optional S3/MinIO model loading.

```bash
# On DGX Spark #1
cp -r spark1/ /opt/vllm
cd /opt/vllm
cp .env.example .env && nano .env    # set MODEL_PATH and tune memory flags
docker compose up -d
```

### 2.3 Deploy vLLM on DGX Spark #2 (Chat)

Same process — use `spark2/` from this repo. It also includes a commented-out backup embedding service for redundancy; see [Redundancy and Failure Handling](#redundancy-and-failure-handling).

```bash
# On DGX Spark #2
cp -r spark2/ /opt/vllm
cd /opt/vllm
cp .env.example .env && nano .env
docker compose up -d
```

### 2.4 Deploy LiteLLM Gateway

LiteLLM runs on any machine on the network — it's lightweight (Python process, no GPU needed). It can run on one of the DGX Sparks or on a separate VM. Use `gateway/` from this repo, which bundles LiteLLM and nginx in a single Docker Compose stack:

```bash
cp -r gateway/ /opt/gateway
cd /opt/gateway
cp .env.example .env && nano .env    # set LITELLM_MASTER_KEY
nano litellm-config.yaml            # replace 10.x.x.10 / 10.x.x.11 with real IPs
nano nginx.conf                     # replace llm.example.com with your domain
docker compose up -d
```

Key files in `gateway/`:

| File | What to configure |
|---|---|
| `.env` | `LITELLM_MASTER_KEY` — used for admin API and generating team keys |
| `litellm-config.yaml` | Backend vLLM URLs, model names, health check interval, failover routing |
| `nginx.conf` | Your domain name; uncomment the HTTPS block if using TLS |
| `docker-compose.yml` | LiteLLM image version; TLS certificate volume mounts (if using HTTPS) |

The master key is loaded from the environment variable `LITELLM_MASTER_KEY` (not hardcoded in the config file).

### 2.5 API Key Management

LiteLLM supports creating per-user or per-team API keys:

```bash
# Create a key for a team (use llm-gateway.internal:4000 or llm.example.com)
curl http://llm-gateway.internal:4000/key/generate \
  -H "Authorization: Bearer sk-change-this-to-a-secure-key" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "data-science",
    "max_budget": 100,
    "models": ["qwen3-embed", "qwen3-chat"]
  }'
```

This returns a key like `sk-abc123...` that users include in their requests. LiteLLM tracks usage per key.

### 2.6 Quick Access (No DNS Required)

Once LiteLLM is running, users can immediately access it by IP and port — no DNS, no reverse proxy, no IT tickets:

```
http://10.x.x.20:4000
```

Or set a hostname via `/etc/hosts` on each client machine:

```
# /etc/hosts on client machines
10.x.x.20  llm-gateway.internal
```

Then use `http://llm-gateway.internal:4000`. This is the fastest way to verify remote access works before investing in a custom domain.

### 2.7 Custom Domain Setup (e.g., `llm.example.com`)

To give users a clean URL like `http://llm.example.com` (no port numbers), you need two things: an internal DNS record and the nginx reverse proxy (already included in `gateway/docker-compose.yml`). **No external requests or public DNS changes are required** — this is entirely within your corporate network.

#### Step 1: Internal DNS Record

Ask your network/IT team to create an internal DNS A record:

```
llm.example.com  →  A  →  10.x.x.20  (IP of the machine running the gateway)
```

This is configured on your company's internal DNS server (Active Directory, Bind, Infoblox, etc.). If your company uses split-horizon DNS for `skhms.com`, the record goes in the internal zone only.

**For testing before DNS is set up**, you can use `/etc/hosts` on client machines:

```
# /etc/hosts
10.x.x.20  llm.example.com
```

#### Step 2: Configure nginx

The nginx container is already part of `gateway/docker-compose.yml` — no need to install nginx separately. Edit `gateway/nginx.conf`:

1. Replace `llm.example.com` with your domain
2. For HTTPS: uncomment the `server { listen 443 ssl; ... }` block and mount your TLS certificate (see the comments in `gateway/docker-compose.yml`)

The nginx config includes `proxy_http_version 1.1` and `proxy_set_header Connection ""` which are required for streaming chat completions to work correctly.

```bash
# After editing nginx.conf, reload without restarting:
cd /opt/gateway && docker compose exec nginx nginx -s reload
```

#### What to ask your IT team

1. "Can you create an internal DNS A record for `llm.example.com` pointing to `10.x.x.20`?"
2. If HTTPS is required: "Can you issue an internal TLS certificate for `llm.example.com`?"

Users then connect to `http://llm.example.com` (or `https://`) with no port numbers.

---

## Part 3: Client Integration

All examples below use the custom domain. If you haven't set that up yet, swap the base URL:

| Setup | Base URL |
|---|---|
| **Quick access (no DNS)** | `http://10.x.x.20:4000/v1` or `http://llm-gateway.internal:4000/v1` |
| **Custom domain (with nginx)** | `http://llm.example.com/v1` |

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Use one of:
#   "http://llm-gateway.internal:4000/v1"  (quick access, no DNS/nginx needed)
#   "http://llm.example.com/v1"             (custom domain with nginx)
BASE_URL = "http://llm.example.com/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="sk-your-team-key",
)

# --- Embeddings ---
embed_response = client.embeddings.create(
    model="qwen3-embed",
    input=["First document to embed", "Second document to embed"],
)
for item in embed_response.data:
    print(f"Embedding dimension: {len(item.embedding)}")

# --- Chat Completion ---
chat_response = client.chat.completions.create(
    model="qwen3-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is retrieval-augmented generation?"},
    ],
    max_tokens=256,
)
print(chat_response.choices[0].message.content)

# --- Streaming Chat ---
stream = client.chat.completions.create(
    model="qwen3-chat",
    messages=[{"role": "user", "content": "Explain transformers briefly."}],
    max_tokens=256,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl

```bash
# Replace llm.example.com with llm-gateway.internal:4000 if using quick access

# Embedding
curl http://llm.example.com/v1/embeddings \
  -H "Authorization: Bearer sk-your-team-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embed", "input": "Hello world"}'

# Chat completion
curl http://llm.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-your-team-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

### JavaScript / TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://llm.example.com/v1",
  apiKey: "sk-your-team-key",
});

// Embedding
const embedding = await client.embeddings.create({
  model: "qwen3-embed",
  input: "Text to embed",
});
console.log(embedding.data[0].embedding.length);

// Chat
const chat = await client.chat.completions.create({
  model: "qwen3-chat",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(chat.choices[0].message.content);
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="qwen3-embed",
    openai_api_base="http://llm.example.com/v1",
    openai_api_key="sk-your-team-key",
)

llm = ChatOpenAI(
    model="qwen3-chat",
    openai_api_base="http://llm.example.com/v1",
    openai_api_key="sk-your-team-key",
)

# These can now be used in any LangChain pipeline (RAG, agents, etc.)
```

---

## Monitoring and Observability

### vLLM Prometheus Metrics

vLLM exposes Prometheus-format metrics at `/metrics` on each node:

```bash
curl http://10.x.x.10:8000/metrics
curl http://10.x.x.11:8000/metrics
```

Key metrics to watch:

| Metric | What It Tells You |
|---|---|
| `vllm:num_requests_running` | Current concurrent requests |
| `vllm:num_requests_waiting` | Requests queued (indicates saturation) |
| `vllm:gpu_cache_usage_perc` | KV cache utilization — if consistently >90%, reduce `max-model-len` or increase memory |
| `vllm:avg_generation_throughput_toks_per_s` | Tokens per second throughput |
| `vllm:e2e_request_latency_seconds` | End-to-end request latency |

### LiteLLM Usage Tracking

LiteLLM logs per-request usage. You can query it:

```bash
# Get usage for a specific key
curl http://llm.example.com/key/info \
  -H "Authorization: Bearer sk-change-this-to-a-secure-key" \
  -d '{"key": "sk-your-team-key"}'
```

### Grafana Dashboard (Optional)

Use `monitoring/` from this repo to deploy Prometheus and Grafana:

```bash
cp -r monitoring/ /opt/monitoring
cd /opt/monitoring
nano prometheus.yml     # replace 10.x.x.10 / 10.x.x.11 with real IPs
docker compose up -d
```

Grafana is available at `http://<host>:3000` (default login: `admin` / `admin`). Add Prometheus as a data source (`http://prometheus:9090`) and build dashboards using the vLLM metrics listed above.

---

## Quantization: Running Larger Models

Quantization reduces the precision of model weights (e.g., FP16 to INT4), dramatically cutting memory usage with minimal quality loss. On DGX Spark's 128 GB unified memory, quantization is what unlocks 70B+ models.

### Why Quantize?

| Model | FP16 Memory | INT4 (GPTQ/AWQ) Memory | Fits on DGX Spark? |
|---|---|---|---|
| Qwen3-8B | ~16 GB | ~4 GB | FP16: Yes, INT4: Yes |
| Qwen3-32B | ~64 GB | ~16 GB | FP16: Yes (tight), INT4: Yes (comfortable) |
| Qwen3-72B | ~144 GB | ~36 GB | FP16: **No**, INT4: **Yes** |
| Qwen3-235B-A22B (MoE) | ~470 GB | ~118 GB | FP16: No, INT4: Yes (tight) |

**INT4 quantization typically loses 1-3% on benchmarks** — for most practical applications (chat, RAG, summarization), users won't notice the difference.

### Quantization Formats Supported by vLLM

| Format | How It Works | Best For |
|---|---|---|
| **AWQ** | Activation-aware quantization — preserves weights that matter most to output quality | Best overall quality at INT4. Recommended for production. |
| **GPTQ** | Post-training quantization using calibration data | Widely available, good quality. Slightly slower than AWQ on some hardware. |
| **FP8** | 8-bit floating point — higher quality than INT4, uses more memory | When you have memory headroom but want speedup over FP16 |
| **BitsAndBytes (NF4)** | Normal-float 4-bit, used primarily for fine-tuning | Fine-tuning with QLoRA. Not ideal for serving. |

### Getting Quantized Models from ModelScope

Many popular models have pre-quantized variants available:

```bash
# AWQ quantized (recommended)
git clone https://www.modelscope.ai/Qwen/Qwen3-72B-AWQ.git /models/Qwen3-72B-AWQ

# GPTQ quantized
git clone https://www.modelscope.ai/Qwen/Qwen3-72B-GPTQ-Int4.git /models/Qwen3-72B-GPTQ-Int4
```

> **Tip:** Always check ModelScope for pre-quantized variants before quantizing yourself. Search for the model name with `-AWQ`, `-GPTQ-Int4`, or `-GPTQ-Int8` suffixes.

### Serving a Quantized Model

vLLM auto-detects the quantization format from the model config:

```bash
docker run --rm --gpus all \
  -v /models/Qwen3-72B-AWQ:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --enforce-eager \
    --gpu-memory-utilization 0.9
```

Note: `--dtype` is not needed — vLLM reads the quantization config from the model's `config.json` or `quantize_config.json`.

### Quantizing a Model Yourself

If a pre-quantized variant isn't available on ModelScope, you can quantize locally using AutoAWQ:

```bash
pip install autoawq
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/models/Qwen3-32B"
quant_path = "/models/Qwen3-32B-AWQ"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize — requires calibration data (a small sample of text)
model.quantize(
    tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4},
)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

> **Warning:** Quantization itself requires loading the full FP16 model into memory. For a 72B model (~144 GB), you cannot quantize on a single DGX Spark. Either quantize on a machine with more memory, or download a pre-quantized variant.

### Recommended Setup with Quantization

With quantization, you can unlock a much stronger architecture:

```mermaid
graph LR
    subgraph "DGX Spark #1"
        E["Qwen3-Embedding-8B<br/>FP16 (~16 GB)"]
        R["Qwen3-Reranker-0.6B<br/>FP16 (~1.2 GB)"]
    end
    subgraph "DGX Spark #2"
        C["Qwen3-72B-AWQ<br/>INT4 (~36 GB)"]
    end

    style E fill:#3498db,stroke:#2980b9,color:#fff
    style R fill:#3498db,stroke:#2980b9,color:#fff
    style C fill:#2ecc71,stroke:#27ae60,color:#fff
```

Node #1 runs the lightweight embedding and reranker models with plenty of headroom. Node #2 dedicates its full 128 GB to the quantized 72B chat model, leaving ~90 GB for KV cache and concurrent requests.

### Multi-Node Tensor Parallelism (Advanced)

For models too large to fit on a single DGX Spark even with quantization, NVIDIA's vLLM container supports **tensor parallelism across two Sparks** connected via a QSFP cable. This pools both nodes' memory (256 GB total) and uses Ray for distributed coordination.

**Prerequisites:**
- Connect the two DGX Sparks with a QSFP cable (high-speed direct link)
- Configure passwordless SSH between the nodes
- Download the Ray cluster deployment script from [build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm)

**On Node 1 (head):**

```bash
export MN_IF_NAME=enp1s0f1np1  # QSFP network interface
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3}

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

**On Node 2 (worker):**

```bash
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>
export VLLM_HOST_IP=$(ip -4 addr show enp1s0f1np1 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

**Serve a model with tensor parallelism across both nodes:**

```bash
docker exec $VLLM_CONTAINER /bin/bash -c \
  'vllm serve meta-llama/Llama-3.3-70B-Instruct \
   --tensor-parallel-size 2 --max_model_len 2048'
```

> **Trade-off:** Multi-node TP enables larger models (e.g., full FP16 70B across 2 nodes) but adds inter-node communication latency. For most use cases, single-node with quantization (INT4 72B on one Spark) will give better throughput. Use multi-node TP when you need full-precision weights or the model simply won't fit on one node even quantized.

---

## Zero-Downtime Model Updates

When a new model version is released (e.g., Qwen3.5), you need to swap it in without dropping user requests.

### Strategy: Blue-Green Deployment on a Single Node

```mermaid
sequenceDiagram
    participant User
    participant LiteLLM
    participant vLLM_Old as vLLM (port 8000)
    participant vLLM_New as vLLM (port 8001)

    Note over vLLM_Old: Serving current model
    User->>LiteLLM: Request
    LiteLLM->>vLLM_Old: Forward
    vLLM_Old-->>User: Response

    Note over vLLM_New: Start new model on different port
    Note over vLLM_New: Verify health + smoke test

    Note over LiteLLM: Update config to point to port 8001
    User->>LiteLLM: Request
    LiteLLM->>vLLM_New: Forward
    vLLM_New-->>User: Response

    Note over vLLM_Old: Stop old container
```

### Step-by-Step

**1. Clone the new model alongside the old one:**

```bash
git clone https://www.modelscope.ai/Qwen/Qwen3.5-8B.git /models/Qwen3.5-8B
```

**2. Reduce the running container's memory allocation, then start the new one:**

The production container runs at `--gpu-memory-utilization 0.9`, which leaves no room for a second instance. You must drop it to 0.4 first — this requires a brief container restart (the model reloads in ~1-3 minutes):

```bash
# Restart the existing container at lower memory utilization
cd /opt/vllm
GPU_MEM_UTIL=0.4 docker compose up -d --force-recreate vllm-chat
```

Then start the new model on a different port:

```bash
docker run -d --name vllm-chat-new --gpus all \
  -v /models/Qwen3.5-8B:/model \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 8192 \
    --enforce-eager --dtype float16 \
    --gpu-memory-utilization 0.4
```

Both instances now share memory (0.4 + 0.4 = 80% of 128 GB = ~102 GB), which is sufficient for two 8B FP16 models (~16 GB each) with KV cache headroom.

**3. Verify the new model works:**

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/model","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
```

**4. Update LiteLLM to point to the new instance:**

Edit `/opt/gateway/litellm-config.yaml` — update the `qwen3-chat` entry to point to the new port:

```yaml
  - model_name: qwen3-chat
    litellm_params:
      model: openai/Qwen3.5-8B
      api_base: http://10.x.x.11:8001/v1    # ← new port
      api_key: unused
```

Restart LiteLLM (takes seconds, much faster than model loading):

```bash
cd /opt/gateway && docker compose restart litellm
```

**5. Stop the old container and reclaim resources:**

```bash
# Stop the Docker Compose-managed instance (lowered to 0.4 in step 2)
cd /opt/vllm && docker compose stop vllm-chat && docker compose rm -f vllm-chat
```

**6. (Optional) Restart the new container at full memory utilization:**

Now that the old container is gone, restart with `--gpu-memory-utilization 0.9` for maximum throughput:

```bash
docker stop vllm-chat-new && docker rm vllm-chat-new
docker run -d --name vllm-chat --gpus all \
  -v /models/Qwen3.5-8B:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:${VLLM_VERSION:-26.02-py3} \
  vllm serve /model \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --enforce-eager --dtype float16 \
    --gpu-memory-utilization 0.9
```

Update LiteLLM config back to port 8000 and restart.

> **Downtime window:** The only user-facing disruption is the LiteLLM restart (~2-3 seconds). The new model is already warm and serving before the switch.

---

## Redundancy and Failure Handling

With only 2 DGX Spark nodes, full high-availability isn't practical — but you can minimize the blast radius of failures.

### What Happens When a Node Goes Down?

| Failure | Impact | Mitigation |
|---|---|---|
| **Spark #1 (embedding) goes down** | RAG pipelines break, no new embeddings | Run a small backup embedding instance on Spark #2 |
| **Spark #2 (chat) goes down** | No chat completions | Users get clear error from LiteLLM, not a timeout |
| **LiteLLM goes down** | All requests fail | Run LiteLLM on both nodes behind a shared DNS |
| **Network issue between LiteLLM and vLLM** | Requests timeout | LiteLLM health checks detect and surface the issue |

### LiteLLM Health Checks

`gateway/litellm-config.yaml` includes `health_check_interval: 30` — LiteLLM polls `GET /health` on each vLLM backend every 30 seconds. When a backend is unhealthy, LiteLLM returns a clear error message immediately instead of making users wait for a TCP timeout.

### Backup Embedding Instance

If your embedding pipeline is critical (e.g., production RAG), run a small backup on the chat node. Both config changes are already in the repo as commented-out blocks:

1. **On Spark #2:** Uncomment `vllm-embedding-backup` in `spark2/docker-compose.yml` and lower `GPU_MEM_UTIL` to `0.7` in `.env` so both services fit (chat at 70% + backup embed at 15% = 85% of 128 GB).

2. **In the gateway:** Uncomment the backup `qwen3-embed` entry in `gateway/litellm-config.yaml`. The `router_settings` block is already configured for latency-based routing with automatic failover (`num_retries: 2`, `allowed_fails: 2`).

LiteLLM will automatically route embedding requests to the backup if the primary is down.

### Simple Uptime Monitoring

`scripts/health_check.sh` polls all endpoints every minute and writes alerts to syslog and a rotating log file. Edit the `ENDPOINTS` array at the top to match your IPs.

```bash
# Install and enable
sudo cp scripts/health_check.sh /opt/monitoring/health_check.sh
sudo chmod +x /opt/monitoring/health_check.sh

# Add to crontab
echo "* * * * * /opt/monitoring/health_check.sh" | crontab -

# Configure log rotation (prevents /var/log/llm-health.log from growing unbounded)
sudo tee /etc/logrotate.d/llm-health > /dev/null <<'EOF'
/var/log/llm-health.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root adm
}
EOF
```

---

## Troubleshooting

### Model files are 130 bytes (LFS pointers, not actual weights)

```bash
cd /models/Qwen3-Embedding-8B
git lfs pull
```

If `git lfs` fails behind a proxy, download `.safetensors` files manually from the ModelScope web UI.

### vLLM crashes with CUDA error on DGX Spark

Ensure you're using:
1. NVIDIA's DGX Spark container (not upstream vLLM)
2. The `--enforce-eager` flag

```bash
# Verify GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv
# Should show sm_121 or similar
```

### Out of memory

- Lower `--gpu-memory-utilization` (e.g., 0.8)
- Lower `--max-model-len` (e.g., 4096) — KV cache size scales with context length
- Lower `--max-num-seqs` (e.g., 64) — caps the number of concurrent in-flight sequences; this is often the most surgical fix when the model itself fits but batching causes OOM
- Use a quantized model variant (INT4/INT8)
- Don't run other GPU workloads alongside vLLM — the 128 GB is unified and shared
- **Flush the Linux buffer cache** — DGX Spark's unified memory architecture means the OS buffer cache competes with the GPU for the same 128 GB. If the model should fit but OOM still occurs, flush the cache:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### LiteLLM can't reach vLLM backends

```bash
# From the LiteLLM host, verify connectivity
curl http://10.x.x.10:8000/health
curl http://10.x.x.11:8000/health
```

Check that Docker isn't binding to `127.0.0.1` — vLLM must listen on `0.0.0.0`.

### Slow inference (~20-30% slower than expected)

This is expected on DGX Spark due to `--enforce-eager`. The sm_121 architecture doesn't yet have full CUDA graph support in vLLM/Triton. Track NVIDIA's container releases for improvements.

---

## Key Takeaways

1. **DGX Spark requires NVIDIA's vLLM container** — upstream vLLM doesn't fully support the sm_121 architecture. Always use `--enforce-eager`.

2. **128 GB unified memory is generous but shared** — an 8B model in FP16 uses ~16 GB, leaving plenty of room. But don't run other heavy workloads alongside vLLM.

3. **Quantization unlocks larger models** — INT4 (AWQ/GPTQ) cuts memory by 75% with 1-3% quality loss. A 72B model that won't fit in FP16 (144 GB) fits comfortably quantized (~36 GB). Always check ModelScope for pre-quantized variants before quantizing yourself.

4. **One model per node for production** — dedicating each DGX Spark to a single model maximizes KV cache memory and concurrent request throughput. With quantization, consider embedding + reranker on node #1, and a large quantized chat model on node #2.

5. **LiteLLM is the right abstraction layer** — it turns multiple vLLM instances into a single API gateway with clean model names, API keys, usage tracking, health checks, and automatic failover between backends.

6. **Plan for model updates** — use blue-green deployment (new model on a different port, verify, switch LiteLLM config, stop old container) to avoid downtime. The only disruption is the LiteLLM restart (~2-3 seconds).

7. **Build in redundancy where it matters** — with 2 nodes you can't have full HA, but running a backup embedding instance on the chat node and configuring LiteLLM's fallback routing handles the most common failure scenarios.

8. **Git LFS is the most common gotcha** — always verify that cloned model files are actual weights (multiple GB), not 130-byte LFS pointers.

9. **Any OpenAI SDK client works** — because both vLLM and LiteLLM expose OpenAI-compatible APIs, users can integrate with Python, TypeScript, curl, LangChain, or any tool that supports the OpenAI API format.

10. **Start with local testing, then go to production** — validate on a single node with the smoke test script before deploying Docker Compose services and the LiteLLM gateway.

## References

### Hardware
1. [NVIDIA DGX Spark Product Page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — Official specs and overview
2. [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) — Detailed hardware documentation
3. [NVIDIA DGX Spark In-Depth Review (LMSYS)](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) — Performance benchmarks and analysis

### vLLM on DGX Spark
4. [vLLM for DGX Spark (NVIDIA)](https://build.nvidia.com/spark/vllm) — NVIDIA's optimized vLLM container, supported models, multi-node setup with Ray, and DGX Spark-specific configuration
5. [vLLM DGX Spark Compatibility Discussion](https://discuss.vllm.ai/t/nvidia-dgx-spark-compatibility/1756) — Community discussion on sm_121 support
6. [vLLM Installation on DGX Spark Guide](https://medium.com/@stablehigashi/vllm-installation-on-dgx-spark-gb10-sm-121-and-qwen-3-5-serving-guide-9eba91e448f8) — Step-by-step source build guide
7. [Building vLLM from Source on DGX Spark](https://medium.com/@anveshkumarchavidi/installing-vllm-on-nvidia-dgx-spark-from-source-4dde137ff3ef) — Alternative source build approach
8. [dgx-spark-vllm-setup (GitHub)](https://github.com/eelbaz/dgx-spark-vllm-setup) — One-command setup script

### vLLM Documentation
9. [vLLM Serve CLI Reference](https://docs.vllm.ai/en/stable/cli/serve/) — Command-line options
10. [vLLM Embedding Documentation](https://docs.vllm.ai/en/v0.7.0/getting_started/examples/embedding.html) — Embedding model serving guide
11. [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) — API compatibility details

### Models
12. [Qwen3-Embedding-8B (Hugging Face)](https://huggingface.co/Qwen/Qwen3-Embedding-8B) — Model card and specifications
13. [Qwen3-Embedding (GitHub)](https://github.com/QwenLM/Qwen3-Embedding) — Source repository and documentation

### LiteLLM
14. [LiteLLM Documentation](https://docs.litellm.ai/) — Proxy configuration, API key management, and usage tracking
15. [LiteLLM GitHub](https://github.com/BerriAI/litellm) — Source code and Docker images
