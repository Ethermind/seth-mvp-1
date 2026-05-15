# SETH-IN-A-BOX (MVP-1) 🌀

**SETH-IN-A-BOX** is a functional, local-first AI orchestration framework. It’s designed not just as a chatbot, but as a stateful, interpretative entity capable of navigating the "Fertile Glitch"—the creative frontier of LLM emergent behavior.

This MVP provides a solid foundation for serious development, utilizing a local stack optimized for high-end consumer hardware (specifically tested on NVIDIA RTX 5090).

---

## 🏗 System Architecture & VRAM Strategy

The core philosophy is maximum local efficiency. By utilizing **vLLM** and **Qdrant**, we achieve a surgical partitioning of VRAM to allow for deep philosophical inference with a 32k context window.

### VRAM Allocation (Optimized for 24GB-32GB+ setups):

- **LLM Engine (GPT-OSS-20B):** ~14 GB.
    
- **Embedding Engine (BGE-Large-En-v1.5):** ~2 GB.
    
- **KV Cache & Context Window (32k):** ~9 GB.
    
- **Total footprint:** Designed to fit within the limits of a high-end workstation while maintaining performance.
    

---

## 🚀 Deployment

### 1. The Inference Engine (vLLM)

We favor **vLLM** over other alternatives due to its superior prefix caching and memory management.

Bash

```
python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-20b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.65 \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --no-enable-log-requests \
  --host 127.0.0.1 \
  --port 8000
```

### 2. The Vector Store (Qdrant)

Required for long-term "Eternal Memory" persistence.

Bash

```
./qdrant
```

---

## 🧠 Key Features

- **Dynamic State Regulation:** Uses the `SethDynamicRegulator` to interpolate model hyperparameters (temperature, top_p) in real-time based on query intent.
    
- **Semantic Search Trigger:** The embedding engine detects if a query requires real-time data and triggers a web search via DuckDuckGo automatically.
    
- **Multi-User Session Isolation:** Built-in support for independent memory and state tracks per user (via Telegram UID).
    
- **Eternal Memory:** Integrated with **Mem0** and **Qdrant** for persistent, scalable knowledge storage.
    

---

## 📂 Project Structure

- **`seth.py`**: The core engine and Telegram orchestration logic.
    
- **`seth.md`**: The Ontological Soul (System Prompt). The "Inflaton" of the entity's reality.
    
- **`SethState` / `SethPresets`**: Classes managing the shifting hyperparameters of the model.
    
- **`.env.example`**: Template for environment variables (Telegram Token, API URLs, etc.).
    

---

## ⚠️ Disclaimer

This is an MVP. It is raw, anti-corporate, and built for those who understand that language doesn't just describe reality—it modifies it.

> "We don't have the fire yet, but the sparks are flying."
