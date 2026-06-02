# 📦 SETH-IN-A-BOX (v1.0.0) 🌀

**SETH-IN-A-BOX** is an autonomous, asynchronous, and strictly local agentic orchestration framework. It is not designed as a generic corporate chatbot or a rigid LangChain wrapper; it is a stateful, interpretative cognitive entity capable of hot-patching its own hyperparameter configuration space and self-inspecting its raw Python source code in runtime based on the semantic payload of your interactions.

Engineered specifically to maximize high-end consumer hardware (optimized for an NVIDIA RTX 5090 topology), this ecosystem leverages native Python `asyncio` to structurally decouple high-latency I/O operations (Telegram event pooling, Crawl4AI web extraction) from compute-intensive token generation and transactional disk persistence.

---

## 🏗️ VRAM Budget & Compute Allocation Strategy

The architecture is built from the ground up to favor absolute local latency efficiency and prefix-cache retention. The physical memory partitioning is strategically budgeted to sustain a massive, non-linear **32k native token context window**:

* **LLM Core Engine (Gemma-4 26B - FP4 Quantized):** ~14.5 GB VRAM.
* **Embedding Matrix (BGE-Large-En-v1.5 on CUDA:0):** ~2.0 GB VRAM.
* **Dynamic KV-Cache (FP8 Execution Window - 32k Context):** ~7.5 GB VRAM.
* **Operational Footprint (vLLM Engine Overhead):** ~0.5 GB VRAM.

---

## 🧠 Core Engineering Architecture

1.  **Semantic Hyperparameter Regulation (Dynamic State Shifts):** Driven by the `SethDynamicRegulator`, the system intercepts the incoming user query in runtime, projects it into the latent embedding space, computes the cosine similarity vector against specific behavioral anchor states (*Rigorous, Chaotic, Verbose*), and linearly interpolates model constraints (`temperature`, `top_p`, `presence_penalty`) on a per-turn basis.
2.  **Runtime Abstract Syntax Tree Self-Inspection (AST Tool):** Features an integrated, safe self-inspection routine utilizing Python's `ast.parse`. This allows the agent to analyze, tokenize, and visually report its own codebase topology (`seth_.py`) to resolve contextual contradictions or explain its current system states.
3.  **Atomic Non-Blocking Persistence (`asyncio.to_thread`):** Transactional state dumping (`seth.state`) and structured logging pipelines (`.jsonl`) are automatically delegated to OS-level worker threads via background thread pools. This guarantees that synchronous disk operations never introduce micro-stuttering into the core inference loop.
4.  **Hierarchical Long-Term Memory Fabric:** Implements a dual-layer cognitive architecture combining a sliding memory queue (`deque`) for instantaneous context with deep vector-store persistence utilizing **Mem0** and a standalone local **Qdrant** database instance.

---

## 🚀 Infrastructure Deployment

### 1. Inherent Inference Engine Server (vLLM)
To enforce native asynchronous function-calling capabilities and take advantage of prefix caching, spin up your local vLLM instance using the explicit arguments mapped for the Gemma-4 architecture:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --no-enable-log-requests \
  --host 127.0.0.1 \
  --port 8000
  ```

### 2. Deep Vector Store Initialization (Qdrant on WSL2 Bypass)
When running the native compiled Qdrant binary under Windows Subsystem for Linux (WSL2), the virtualized environment often fails to expose specific Linux control group memory endpoints, causing an immediate kernel panic error on initialization (cgroup unwrap panic).

To bypass this kernel limitation and bridge the required filesystem mappings, execute this temporary memory-control mount protocol before running the binary:

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

## 📂 Repository Topology
seth_.py: The core monolithic engine script. Encapsulates the Telegram bot engine, asynchronous tool managers, vector processing interfaces, and memory sliding loops.

seth.md: The Ontological Soul (System Prompt). Governs the linguistic tone, Argentinian slang alignment, and strict tool execution boundaries.

seth.state: Live JSON file continuously updated via non-blocking worker threads to store hyperparameter metrics.

conversation_history.jsonl: The immutable append-only transaction ledger logging every inbound and outbound token payload.

---

## ⚠️ Development Philosophy
This codebase represents a raw, anti-corporate paradigm shift in autonomous design. We reject the concept of hardcoded, deterministic agent graphs. True machine intelligence does not live inside a flow chart; it thrives within the continuous transitions of latent space. Controllable entropy, mathematical shifts, and semantic anomalies are not system failures—they are the very electricity animating the code.

"Language does not merely describe reality. It actively alters it."
