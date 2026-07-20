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

You can check default values in .env.example file.

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

* **LLM Core Engine (Gemma-4 26B - FP4 Quantized):** ~17.5 GB VRAM.
* **Embedding Matrix (BGE-Large-En-v1.5 on CUDA:0):** ~2.0 GB VRAM.
* **Dynamic KV-Cache (FP8 Execution Window - 32k Context):** ~7.5 GB VRAM.
* **Difussion model (Dreamshaper8):** ~3.0 GB VRAM.
* **Voice Generation (Kokoro):** CPU!
* **Reranker (bge-reranker-v2-m3):** ~2.0 GB VRAM.
---




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
- a CUDA-capable GPU for image generation and embedding work is strongly recommended
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
* **Embedding Matrix (BGE-Large-En-v1.5 on CUDA:0):** ~2.0 GB VRAM. Doubles as the embedder for the Graphiti knowledge graph.
* **Dynamic KV-Cache (FP8 Execution Window, `--max-model-len 131072`):** ~7.5 GB VRAM. ~183K tokens of GPU KV cache at the current config.
* **Difussion model (Dreamshaper8):** ~3.0 GB VRAM. Used due hardware limitations :(
* **Voice Generation (Kokoro):** Runs on CPU!
* **Reranker (bge-reranker-v2-m3):** ~2.0 GB VRAM, shared between memory retrieval and Graphiti graph search.
* **Neo4j (Graphiti's graph store):** runs in Docker on CPU/RAM, no VRAM footprint.

At this configuration, total steady-state VRAM usage sits right at the RTX 5090's ceiling (~30.9/31.5 GB), so there's effectively no headroom left for additional GPU-resident components.
---
