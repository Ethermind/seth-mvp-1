# SETH-IN-A-BOX

SETH-IN-A-BOX is a local, multimodal Telegram agent implemented around the main runtime in seth_poc.py. It is intended as a technical prototype for experimenting with local LLM orchestration, tool use, per-user memory, and media generation without depending on external cloud services for the core inference path.

The current implementation supports:

- receiving text, voice, and image messages through Telegram
- transcribing audio locally with Whisper
- sending text responses and, when appropriate, generated media attachments
- generating images with a local diffusion model
- synthesizing speech with Kokoro
- maintaining short-term per-user history and long-term memory through Mem0 and Qdrant

## Purpose

This repository is a compact local agent stack intended for experimentation, debugging, and iterative development around:

- tool calling from an LLM
- multimodal message handling
- local model serving
- user-scoped memory
- simple access control for Telegram usage

## Architecture overview

The runtime is organized around a small set of clearly defined components:

- seth_poc.py: main entrypoint and orchestration layer
- seth.md: system prompt used by the bot
- seth.state: persisted inference state
- conversations/: per-user short-term conversation logs
- storage/: generated images, audio, logs, and access-control state

## Runtime requirements

A functional setup requires:

- Python 3.10 or newer
- a CUDA-capable GPU for image generation and embedding work is strongly recommended
- a local OpenAI-compatible LLM endpoint such as vLLM
- a local Whisper-compatible transcription endpoint
- a local Qdrant instance for vector memory
- a Telegram bot token and a registration token

## Environment variables

The bot reads configuration from environment variables and defaults where appropriate. At minimum, the following are expected for normal execution:

- TELEGRAM_TOKEN
- REGISTRATION_TOKEN
- LLM_MODEL
- VLLM_URL
- WHISPER_URL
- WHISPER_MODEL
- IMAGE_MODEL
- EMBEDDING_MODEL
- QDRANT_HOST
- QDRANT_PORT
- API_KEY

## Running the bot

Install the required dependencies, then start the bot with:

```bash
python seth_poc.py
```

The bot initializes its tools, memory layers, and Telegram interface. Access is initially restricted and can be enabled using the registration token configured in REGISTRATION_TOKEN.

## Example services (how I use it)

### vLLM

Example command to start an OpenAI-compatible local inference server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model abhishekchohan/gemma-4-26B-A4B-it-abliterated-AWQ \
  --tool-call-parser gemma4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 45056 \
  --gpu-memory-utilization 0.94 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --no-enable-log-requests \
  --host 127.0.0.1 \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --chat-template-content-format openai \
  --reasoning-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja
```

### Whisper

Example command to expose a local transcription endpoint:

```bash
faster-whisper-server --port 8010 large-v3
```

### Qdrant

Example command to run a local Qdrant instance:

```bash
# 1. Create the system control group directory structure
sudo mkdir -p /sys/fs/cgroup

# 2. Mount a temporary transactional filesystem to bypass kernel locks
sudo mount -t tmpfs tmpfs /sys/fs/cgroup

# 3. Simulate the expected high-watermark memory threshold metrics
echo "max" | sudo tee /sys/fs/cgroup/memory.high

# 4. Grant absolute read/write permissions to prevent operational crashes
sudo chmod 777 /sys/fs/cgroup/memory.high

# 5. Boot up the high-performance vector store instance
./qdrant
```

---

## VRAM Budget & Compute Allocation Strategy

The architecture is built from the ground up to favor absolute local latency efficiency and prefix-cache retention. The physical memory partitioning is strategically budgeted to sustain a massive, non-linear **32k native token context window**:

* **LLM Core Engine (Gemma-4 26B - FP4 Quantized):** ~14.5 GB VRAM.
* **Embedding Matrix (BGE-Large-En-v1.5 on CUDA:0):** ~2.0 GB VRAM.
* **Dynamic KV-Cache (FP8 Execution Window - 32k Context):** ~7.5 GB VRAM.
* **Operational Footprint (vLLM Engine Overhead):** ~0.5 GB VRAM.
* **Difussion model (SD1.5):** ~3.0 GB VRAM.

---