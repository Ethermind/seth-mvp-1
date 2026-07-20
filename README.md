# SETH-IN-A-BOX
###### "Inspired by my prompt engineering research and publications on Medium: https://medium.com/@luis.capra"

SETH-IN-A-BOX is a local, multimodal Telegram agent implemented around the main runtime in seth_poc.py. It is intended as a technical prototype for experimenting with local LLM orchestration, tool use, per-user memory, and media generation without depending on external cloud services for the core inference path.

The current implementation supports:

- receiving text, voice, and image messages through Telegram
- transcribing audio locally with Whisper
- sending text responses and, when appropriate, generated media attachments
- generating images with a local diffusion model
- synthesizing speech with Kokoro
- maintaining short-term per-user history and long-term memory through Mem0 and Qdrant
- a relational/temporal knowledge graph layer (Graphiti + Neo4j) that shadow-writes every conversation turn in the background and is queryable through an explicit tool for relationship and "how did this change over time" style questions
- live web search with DuckDuckGo (DDGS) plus concurrent page crawling (crawl4ai) for real, up-to-date page content instead of just snippets

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

Memory is split across two complementary layers: Mem0 + Qdrant handle semantic fact retrieval ("what do I know about X"), while Graphiti + Neo4j handle relational and temporal reasoning ("how are X and Y connected", "how did this change over time"). Graph writes happen asynchronously in shadow mode on every turn (fire-and-forget, never blocking the response), and are only surfaced back to the model through an explicit `query_relationship_graph` tool rather than always-on context injection — entity extraction for the graph runs through a lightweight GLiNER2-based client, with a BGE reranker used for graph search results.

## Runtime requirements

A functional setup requires:

- Python 3.10 or newer
- a CUDA-capable GPU for image generation and embedding work is strongly recommended (verified on driver 610.43.02 / CUDA 13.0 toolkit, running torch's cu128 build — the driver is backwards compatible, so cu128 wheels work fine even on newer driver/toolkit versions)
- a local OpenAI-compatible LLM endpoint such as vLLM
- a local Whisper-compatible transcription endpoint
- a local Qdrant instance for vector memory
- a local Neo4j instance for the knowledge graph (Graphiti)
- a Telegram bot token and a registration token

## Development platform

```bash
╔════════════════════════════════════════════════════╗
║           SETH AI DEVELOPMENT PLATFORM             ║
╠════════════════════════════════════════════════════╣
║ CPU          : AMD Ryzen 9 9950X3D                 ║
║ RAM          : 128 GB DDR5-6000 G.Skill            ║
║ Primary GPU  : ASUS ROG Astral RTX 5090 32 GB      ║
║ Secondary GPU: NVIDIA RTX 3050 6 GB                ║
║ Storage      : 3 × Samsung 990 Pro 4 TB NVMe       ║
║ Mainboard    : ASUS ProArt X870E-Creator WiFi      ║
║ PSU          : Seasonic PRIME TX-1600 Titanium     ║
║ Cooling      : Arctic Liquid Freezer III Pro 420   ║
║ Chassis      : Fractal Design Define 7 XL          ║
╚════════════════════════════════════════════════════╝
```

The dev box runs Ubuntu under WSL2. Beyond the `.env` file consumed by `seth_poc.py`, a few shell-level variables are needed for the CUDA toolchain and vLLM to behave correctly on the RTX 5090 (Blackwell / `sm_120`) alongside the older RTX 3050. Export these in your `.bashrc` (or an activation script for your conda env) before building anything from source or launching vLLM:

```bash
# CUDA toolkit used to build torch/vLLM extensions (matches the installed cu128 torch build)
export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"

# Required for the RTX 5090's Blackwell architecture — without this, anything that
# compiles CUDA kernels from source (flash-attn, punica kernels, etc.) will target
# the wrong architecture and fail or silently underperform.
export TORCH_CUDA_ARCH_LIST="12.0"

# Keep GPU indices stable and consistent with nvidia-smi's ordering
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1"   # 0 = RTX 5090, 1 = RTX 3050

# Dual-GPU of mismatched generations (5090 + 3050) can hit NCCL P2P transport issues
export NCCL_P2P_DISABLE="1"

# vLLM-specific: pins the legacy engine and enables punica kernels for LoRA-style ops
export VLLM_USE_V1="0"
export VLLM_INSTALL_PUNICA_KERNELS="1"

# Parallel build jobs when compiling extensions from source
export MAX_JOBS="8"
```

Note: `CUDA_HOME` points at the 12.8 toolkit here even though `nvcc --version` may report a newer release (13.0) if you have multiple toolkits installed side by side — what matters is that it matches the CUDA build your installed torch wheel was compiled against (check with `python -c "import torch; print(torch.version.cuda)"`), not the newest toolkit on disk.

## Environment variables

The bot reads configuration from environment variables and defaults where appropriate. The `_env.example` file covers the core variables needed for a normal run:

- TELEGRAM_TOKEN
- REGISTRATION_TOKEN
- LLM_MODEL
- VLLM_URL
- EMBEDDING_MODEL
- EMBEDDING_MODEL_DIMS
- QDRANT_HOST
- QDRANT_PORT
- WHISPER_URL
- WHISPER_MODEL
- IMAGE_MODEL

A few additional variables exist in code with sane defaults and don't need to be set unless you're customizing them:

- API_KEY (defaults to `NONE`, used as the vLLM/OpenAI-compatible API key)
- MAX_TOKENS (defaults to `32768`)
- NEO4J_URI (defaults to `bolt://localhost:7687`)
- NEO4J_USER (defaults to `neo4j`)
- NEO4J_PASSWORD (defaults to empty, set this to match your Neo4j container's `NEO4J_AUTH`)

You can check default values in the `_env.example` file.

## Running the bot

1. Clone the repo and create a virtual environment:

   ```bash
   git clone https://github.com/<your-username>/seth-in-a-box.git
   cd seth-in-a-box
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install PyTorch first, matching your CUDA driver (verified working with the cu128 build even on driver 610.43.02 / CUDA 13.0, since NVIDIA drivers stay backwards compatible — check the [PyTorch install matrix](https://pytorch.org/get-started/locally/) if your setup differs):

   ```bash
   pip install torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

   To check your own driver/CUDA situation before picking a build:

   ```bash
   nvidia-smi                 # driver version + max CUDA it supports
   nvcc --version              # installed CUDA toolkit, if any
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
   ```

   The `CUDA Version` shown by `nvidia-smi` is the *maximum* your driver supports, not necessarily the build torch is actually using — trust the `torch.version.cuda` output over the other two.

3. Install the rest of the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Kokoro also needs the `espeak-ng` system package:

   ```bash
   sudo apt-get install espeak-ng
   ```

4. Copy `_env.example` to `.env` and fill in your tokens/URLs:

   ```bash
   cp _env.example .env
   ```

5. Make sure the external services are up before starting the bot: vLLM, the Whisper endpoint, Qdrant, and Neo4j (see [Example services](#example-services-how-i-use-it) below).

6. Start the bot:

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

NOTE: I dont recommend the use of /Systran/faster-whisper-medium .. it's not reliable.

```bash
faster-whisper-server --port 8010 large-v3
```

### Your Qdrant (how it should be!)

```bash
./qdrant
```

### MY! Qdrant (with quick and dirty fix for WSL2)

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

### NEO4J (using docker.. hate that..)

```bash
docker run -d \
  --name seth-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -v /neo4j_logs:/logs \
  -v /neo4j_data:/data \
  -e NEO4J_AUTH=USER/PASSWORD \
  neo4j:5.26-community
```

---

## VRAM Budget & Compute Allocation Strategy

The architecture is built from the ground up to favor absolute local latency efficiency and prefix-cache retention. The physical memory partitioning is strategically budgeted to sustain a massive, non-linear **128k native token context window**:

* **LLM Core Engine (Gemma-4 26B - AWQ Quantized):** ~17.5 GB VRAM.
* **Embedding Matrix (BGE-Large-En-v1.5):** ~2.0 GB VRAM. Doubles as the embedder for the Graphiti knowledge graph.
* **Dynamic KV-Cache (FP8 Execution Window, `--max-model-len 131072`):** ~7.5 GB VRAM. ~183K tokens of GPU KV cache at the current config.
* **Difussion model (Dreamshaper8 on non dedicated CUDA:1):** ~3.0 GB VRAM. Used due hardware limitations :(
* **Voice Generation (Kokoro):** Runs on CPU!
* **Reranker (bge-reranker-v2-m3):** ~2.0 GB VRAM, shared between memory retrieval and Graphiti graph search.
* **Neo4j (Graphiti's graph store):** runs in Docker on CPU/RAM, no VRAM footprint.

At this configuration, total steady-state VRAM usage sits right at the RTX 5090's ceiling (~30.9/31.5 GB), so there's effectively no headroom left for additional GPU-resident components.
---