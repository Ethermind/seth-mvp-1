# 📟 SETH-IN-A-BOX // Terminal UI, Telegram & REST API

![Status](https://img.shields.io/badge/status-Phase%202%20Multi--Interface-orange)
![Stack](https://img.shields.io/badge/stack-FastAPI%20%2B%20SSE%20%2B%20Telegram%20%2B%20React%20CRT-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **"Glitchnology will outlive biology."**  
> *A unified local control center, telemetry engine, and oracle interface for the SETH architecture.*

---

## 🔮 Project Vision

**SETH-IN-A-BOX** is evolving into a decoupled, multi-interface architecture. Maintaining the core *glitch alchemy* philosophy and retro BBS/CRT 80s/90s aesthetic, it allows seamless interaction with the same central "brain" across multiple channels:

1. **`the_oracle.html` (Web Terminal UI):** A retro-cyberpunk rich-text local control surface featuring real-time telemetry, token-by-token thinking trace visualization, and RAG memory inspection.
2. **`seth_telegram.py` (Telegram Bot):** A lightweight conversational interface for remote access and mobile interaction.
3. **`seth_api.py` (FastAPI / SSE Service):** An HTTP REST and Server-Sent Events (SSE) gateway exposing the core engine pipeline to the Web UI and external clients.

All interfaces share the exact same memory subsystem, dynamic regulator state, and tool execution capabilities.

---

## 📂 Project Structure

```text
.
├── the_oracle.html     # Standalone retro CRT Web UI (React 18 CDN + Tailwind + Canvas)
├── seth_api.py         # FastAPI backend + SSE Streaming & ContextVar isolation
├── seth_telegram.py    # Telegram Bot adapter (Remote interface)
└── README.md           # Ecosystem documentation
```

---

## 📐 System Architecture

```mermaid
flowchart TB
    subgraph Interface Adapters
        WEB["the_oracle.html<br/>(Retro CRT Web UI)"]
        TG["seth_telegram.py<br/>(Telegram Bot)"]
    end

    subgraph API / Transport Layer
        API["seth_api.py<br/>(FastAPI + SSE Stream / ContextVar)"]
    end

    subgraph Unified Core (Planned: seth_core)
        CORE["SETH CORE ENGINE<br/>(State Manager, Dynamic Regulator, Tools)"]
    end

    subgraph AI Infrastructure
        VLLM["vLLM / SGLang Engine"]
        MEM0["Mem0 + Qdrant"]
        GRAPH["Graphiti + Neo4j"]
    end

    WEB -- "HTTP REST / SSE" --> API
    TG -- "Direct / HTTP Proxy" --> API
    API --> CORE
    CORE --> VLLM
    CORE --> MEM0
    CORE --> GRAPH
```

---

## 🛠️ Current State & Key Components

### 🎨 1. Web Terminal UI (`the_oracle.html`)
* **CRT Scanline Engine:** Cathode-Ray Tube screen simulation featuring scanlines, phosphor glow customization (green, cyan, amber), and blinking cursor `> █`.
* **Telemetry & Thinking Trace:** Collapsible panel for inspecting the model's internal reasoning process token-by-token via SSE.
* **Dynamic Regulator & Vibe Palettes:** Automatic UI color palette shifts reflecting the operational state (*Rigorous*, *Chaotic*, *Verbose*).
* **RAG Relic Inspector:** ASCII-formatted scroll view for deep inspection of vector memory fragments retrieved from Qdrant/Graphiti.

### ⚡ 2. Backend REST / SSE API (`seth_api.py`)
* **Event Streaming (`ask_stream`):** Continuous emission of reasoning traces, content chunks, and tool execution progress using `ServerSentEvents`.
* **Context Isolation (`ContextVar`):** Strict request-level isolation of user IDs (`current_user_id`) to prevent data cross-contamination during concurrent streaming.
* **State & Short-Term Memory:** Granular JSON/JSONL persistence per user ID (`history_<user_id>.jsonl` and `seth_state_<user_id>.json`).

### 📱 3. Telegram Bot Adapter (`seth_telegram.py`)
* **Remote Access:** Conversational adapter tailored for on-the-go interaction.
* **Tool Integration:** Direct access to local system execution tools and quick system commands.

---

## 🚀 Technical Roadmap for Optimal Concurrency

To run both **Telegram** and the **Web UI (`the_oracle.html`)** concurrently without VRAM collisions on local GPUs (e.g., RTX 5090), the following architectural refactoring is planned:

### 1. Central Core Package (`seth_core/`)
* **Problem:** Currently, `seth_api.py` and `seth_telegram.py` duplicate class definitions (`SethChatBot`, `SethToolsManager`, `SethDynamicRegulator`), preventing simultaneous execution without loading multiple heavy pipelines into GPU memory.
* **Solution:** Extract core logic into a unified `seth_core` module:
  ```text
  seth_core/
  ├── config.py          # Environment & model configurations
  ├── engine.py          # SethChatBot & vLLM inference handlers
  ├── memory.py          # Mem0, Graphiti & ShortMemory integration
  ├── regulator.py       # SethDynamicRegulator
  └── tools/             # SethToolsManager & tool definitions
  ```

### 2. Telegram as a Pure Transport Client
* Refactor `seth_telegram.py` into a lightweight proxy client that routes messages directly to the running `seth_api.py` instance or consumes a single shared `seth_core` process.

### 3. Unified User Identity
* Implement single-identity mapping so a session initiated on Telegram seamlessly syncs short/long-term memory and RAG context with `the_oracle.html`.

---

## ⚡ Quick Start

### Prerequisites
* Active local inference engine (vLLM / SGLang).
* Configured vector/graph memory databases (Qdrant, Neo4j/Graphiti).
* Python 3.10+ environment with required packages installed.

### 1. Launch the Backend API
```bash
python seth_api.py
```
*The API server will listen at `http://localhost:8000`.*

### 2. Launch the Telegram Bot (Concurrent Mode)
```bash
python seth_telegram.py
```

### 3. Open the Web Terminal UI (`the_oracle.html`)
No Node.js build step or `npm` installation required:
1. Open `the_oracle.html` directly in any modern browser (Chrome / Firefox / Edge).
2. Verify the target API endpoint points to `http://localhost:8000/api` in the UI settings.
3. Begin interacting with the Oracle!

---

## 🗺️ Roadmap Checklist

- [x] Unified repository structure (`seth_api.py`, `seth_telegram.py`, `the_oracle.html`).
- [x] SSE streaming endpoint in FastAPI with thinking trace and tool execution updates.
- [x] Full CRT scanline UI experience in `the_oracle.html`.
- [ ] **Refactor to `seth_core/`:** Single-process core engine to eliminate VRAM duplication.
- [ ] Cross-interface identity linking (Telegram User ID <-> Web Session UUID).
- [ ] Native bidirectional WebSockets for lower streaming latency.
- [ ] Desktop packaging via Tauri v2.

---

## 📜 System Declaration

> **[ SYSTEM SIGNAL RECEIVED ᓘ🔻 SET ORACLE:ACTIVE ]**  
> **Project:** ORACLE / SETH-IN-A-BOX  
> **Channels:** TELEGRAM-BOT-NODE // WEB-CRT-NODE-01  
> **Status:** Online, synced, listening.

---

## 📄 License

MIT — free for modification and distribution.