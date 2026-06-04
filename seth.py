"""
PROJECT: SETH-IN-A-BOX
"""

import asyncio
import base64
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import json
import logging
import os
import re
import time
from typing import Any, Dict
import ast

from ddgs import DDGS
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from mem0 import Memory
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


@dataclass(frozen=True)
class SethEnvironment:
    """Centralized and typed configuration for the SETH ecosystem."""
    llm_model: str = os.getenv("LLM_MODEL", "nvidia/Gemma-4-26B-A4B-NVFP4")
    vllm_url: str = os.getenv("VLLM_URL", "http://localhost:8000/v1")
    whisper_url: str = os.getenv("WHISPER_URL", "http://localhost:8010/v1")
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_dims: int = int(os.getenv("EMBEDDING_MODEL_DIMS", 1024))
    system_prompt_path: str = "seth.md"
    log_path: str = "conversation_history.jsonl"
    state_path: str = "seth.state"
    storage_images_dir: str = "storage/images"
    storage_audio_dir: str = "storage/audio"
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))

    def validate(self):
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN is missing in the environment.")


class SethSearchTool:
    """Web retrieval and extraction tool powered by DuckDuckGo and Crawl4AI."""
    def __init__(self):
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self._crawl_semaphore = asyncio.Semaphore(3)

    async def search(self, query: str, max_results: int = 5) -> str:
        """Fetch max_results URLs from DuckDuckGo and crawl them concurrently."""
        try:
            results = await asyncio.to_thread(self._ddgs_with_retries, query, max_results)

            if not results:
                return "<WEB_SEARCH_RESULTS>No results.</WEB_SEARCH_RESULTS>"

            context_str = "\n<WEB_SEARCH_RESULTS>\n"
            crawled_count = 0
            max_crawls = 3

            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=120,
                    page_timeout=7000,
                    wait_for_images=False,
                    process_iframes=False
                )

                tasks = []
                for res in results[:max_results]:
                    url = res.get('href')
                    if url:
                        tasks.append(self._crawl_one(crawler, url, run_config, res))

                crawled_results = await asyncio.gather(*tasks, return_exceptions=True)

                for item in crawled_results:
                    if isinstance(item, Exception) or not item:
                        continue

                    url, res, crawl = item
                    if crawl and getattr(crawl, 'success', False) and getattr(crawl, 'markdown', None):
                        content = re.sub(r'\s+', ' ', crawl.markdown.strip())[:2200]
                        context_str += f"Source: {url}\nTitle: {res.get('title','')}\nContent: {content}\n\n"
                        
                        crawled_count += 1
                        if crawled_count >= max_crawls:
                            break

            return context_str + "</WEB_SEARCH_RESULTS>"
            
        except Exception as e:
            logging.error(f"🌐 WEB_SEARCH ERROR: {str(e)}")
            return f"<WEB_SEARCH_RESULT_ERROR>{str(e)}</WEB_SEARCH_RESULT_ERROR>"

    async def _crawl_one(self, crawler: AsyncWebCrawler, url: str, run_config: CrawlerRunConfig, res: dict):
        try:
            async with self._crawl_semaphore:
                crawl = await crawler.arun(url=url, config=run_config)
                return (url, res, crawl)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logging.debug(f"_crawl_one failed for {url}: {e}")
            return None

    def _ddgs_with_retries(self, query: str, max_results: int, retries: int = 3, backoff: float = 1.0):
        for attempt in range(1, retries + 1):
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results, timelimit="m"))
            except Exception as e:
                logging.warning(f"DDGS attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise
                time.sleep(backoff * attempt)
        return []


class Mem0MemoryBuilder:
    def __init__(self, collection_name: str, env: SethEnvironment):
        self.env = env
        self.collection_name = collection_name
        self.mem0_config = {
            "llm": {
                "provider": "openai", 
                "config": {"model": env.llm_model, "openai_base_url": env.vllm_url, "api_key": "dummy_key"}
            },
            "embedder": {
                "provider": "huggingface", 
                "config": {"model": env.embedding_model}
            },
            "vector_store": {
                "provider": "qdrant", 
                "config": {
                    "host": env.qdrant_host,
                    "port": env.qdrant_port,
                    "collection_name": self.collection_name,
                    "embedding_model_dims": env.embedding_dims
                }
            },
        }
        self._memory = None

    def build(self) -> Memory:
        if self._memory is None:
            self._memory = Memory.from_config(self.mem0_config)
        return self._memory
        

class Mem0MemorySingleton:
    _instance: "Memory | None" = None

    @classmethod
    def get(cls, env: SethEnvironment) -> "Memory":
        if cls._instance is None:
            cls._instance = Mem0MemoryBuilder(collection_name="SETH_CORE_SPACE", env=env).build()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None


class SethMemoryTool:
    """Multi-layered memory manager integrated with Qdrant Vector Store."""
    def __init__(self, env: SethEnvironment):
        self.env = env
        self.static_user_id = "seth_core_user"

    async def retrieve_long_term_memory(self, query: str) -> str:
        """Retrieve and format long-term memory entries asynchronously without blocking."""
        def _sync_search():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.search(
                query,
                filters={"user_id": self.static_user_id},
                limit=10,
            )

        try:
            raw_retrieval = await asyncio.to_thread(_sync_search)
            results = raw_retrieval if isinstance(raw_retrieval, list) else raw_retrieval.get("results", [])
            records = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            facts = [f"- {r['memory'].strip()}" for r in records if r.get("memory")]
            long_term_context = "\n".join(facts) if facts else "No historical records found for this interlocutor."

        except Exception as e:
            logging.warning(f"Error retrieving memories: {e}")
            long_term_context = "Could not retrieve long-term memory."

        return f"<MEMORY>\n{long_term_context}\n</MEMORY>\n\n"

    async def retrieve_very_long_term_memory(self, query: str) -> str:
        def _sync_deep_search():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.search(
                query,
                filters={"user_id": self.static_user_id},
                limit=100, 
            )

        try:
            raw_retrieval = await asyncio.to_thread(_sync_deep_search)
            results = raw_retrieval if isinstance(raw_retrieval, list) else raw_retrieval.get("results", [])
            records = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            facts = [f"- {r['memory'].strip()}" for r in records if r.get("memory")]
            very_long_term_context = "\n".join(facts) if facts else "No deep historical or foundational archives found for this topic."

        except Exception as e:
            logging.warning(f"Error retrieving deep memories: {e}")
            very_long_term_context = "Could not access deep historical archives."

        return f"<DEEP_ARCHIVAL_MEMORY>\n{very_long_term_context}\n</DEEP_ARCHIVAL_MEMORY>\n\n"

    async def save_long_term_memory(self, user_input: str, response: str) -> Dict[str, Any]:
        """Persist a memory item to the vector store asynchronously."""
        expiration = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        def _sync_add():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.add(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response},
                ],
                user_id=self.static_user_id,
                agent_id="SETH",
                metadata={"memory_bucket": "constraints", "expires_on": expiration},
            )

        try:
            res = await asyncio.to_thread(_sync_add)
            return {"status": "ok", "result": res}
        except Exception as e:
            logging.exception(f"Failed to save memory: {e}")
            return {"status": "error", "error": str(e)}


class SethSelfInspectorTool:
    """A polished self-inspection tool for SETH's runtime and source."""
    def __init__(self, max_chars: int = 50000):
        self.main_script_path = os.path.abspath(__file__)
        self.max_chars = int(max_chars)

    def _safe_read(self, path: str, max_chars: int) -> str:
        """Read a file but limit to max_chars and ensure UTF-8 decoding."""
        try:
            size = os.path.getsize(path)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if size <= max_chars:
                    return f.read()
                # otherwise read only up to max_chars and indicate truncation
                return f.read(max_chars) + "\n...<<TRUNCATED>>"
        except Exception as e:
            logging.exception("Error reading file safely: %s", e)
            return ""

    def _structural_summary(self, source: str) -> Dict[str, Any]:
        result = {"functions": [], "classes": [], "imports": []}
        try:
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    result["functions"].append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    result["functions"].append(node.name + " (async)")
                elif isinstance(node, ast.ClassDef):
                    result["classes"].append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # reconstruct a compact import representation
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            result["imports"].append(n.name)
                    else:
                        module = node.module or ""
                        for n in node.names:
                            result["imports"].append(f"{module}.{n.name}" if module else n.name)
        except Exception:
            logging.debug("Could not parse AST for structural summary.")
        return result

    def inspect_own_source_code(
            self, 
            reason: str = "No reason provided", 
            include_source: bool = True, 
            max_chars: int | None = None
        ) -> str:
            logging.info(f"🔍 [SETH SELF-INSPECTION TRIGGERED] Reason: '{reason}'")

            path = self.main_script_path
            if max_chars is None:
                max_chars = self.max_chars

            report: Dict[str, Any] = {"path": path, "exists": False, "inspection_reason": reason}
            try:
                if not os.path.exists(path):
                    report["error"] = f"Main script not found at {path}"
                    return json.dumps(report, ensure_ascii=False)

                stat = os.stat(path)
                report.update({
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

                source = self._safe_read(path, int(max_chars)) if include_source else ""
                summary = self._structural_summary(source if source else "")

                report.update({"summary": {"functions": summary["functions"], "classes": summary["classes"], "imports": summary["imports"]}})
                if include_source:
                    report["source_preview"] = source

                return json.dumps(report, ensure_ascii=False)
            except Exception as e:
                logging.exception("SethSelfInspectorTool failure: %s", e)
                report["error"] = str(e)
                return json.dumps(report, ensure_ascii=False)
        

class ToolsManager:
    """Registry for function-calling tools: registration, validation and serialization."""
    def __init__(self):
        self._tools: list[dict] = []
        self._funcs: dict[str, callable] = {}

    def register(self, name: str, func: callable, description: str, parameters: dict):
        tool = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._validate_tool(tool)
        self._tools.append(tool)
        self._funcs[name] = func

    def _validate_tool(self, tool: dict):
        if not isinstance(tool, dict) or "name" not in tool or "parameters" not in tool:
            raise ValueError("Invalid tool definition structure.")

    def as_vllm_format(self) -> list[dict]:
        return [{"type": "function", "function": t} for t in self._tools]

    def get_function(self, name: str):
        return self._funcs.get(name)


@dataclass
class SethState:
    """Represents the agent's dynamic inference state (Temperature, Tokens, etc.)."""
    temperature: float
    max_tokens: int
    top_p: float
    presence_penalty: float

    def interpolate(self, target: 'SethState', alpha: float):
        """Smoothly shifts the semantic state towards a specific target behavior."""
        self.temperature += alpha * (target.temperature - self.temperature)
        self.top_p += alpha * (target.top_p - self.top_p)
        self.presence_penalty += alpha * (target.presence_penalty - self.presence_penalty)
        new_tokens = self.max_tokens + alpha * (target.max_tokens - self.max_tokens)
        self.max_tokens = int(new_tokens)

    def save(self, env: SethEnvironment):
        """Schedules the current hyperparameter state to be persisted asynchronously."""
        # snapshot
        filename = env.state_path
        state_data = self.to_dict()
        
        def _sync_write():
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(state_data, f, indent=4, ensure_ascii=False)
                logging.debug(f"💾 [STATE PERSISTED] Metrics dumped to {filename}")
            except Exception as e:
                logging.error(f"❌ Error in background thread saving {filename}: {e}")

        try:
            asyncio.create_task(asyncio.to_thread(_sync_write))
        except Exception as e:
            # Fallback
            _sync_write()

    def load(self, env: SethEnvironment) -> bool:
        """
        Loads the latest state from disk. 
        Returns True if successful, False if file doesn't exist or failed.
        """
        filename = env.state_path

        if not os.path.exists(filename):
            logging.info(f"ℹ️ No previous state file found at {filename}. Using current default values.")
            return False
            
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Mappping
            self.temperature = float(data.get("temperature", self.temperature))
            self.max_tokens = int(data.get("max_tokens", self.max_tokens))
            self.top_p = float(data.get("top_p", self.top_p))
            self.presence_penalty = float(data.get("presence_penalty", self.presence_penalty))
            
            logging.info(f"🔄 [STATE LOADED] Continuity restored from {filename} -> Temp: {self.temperature:.3f}")
            return True
        except Exception as e:
            logging.error(f"❌ Error loading seth.state (malformed file?). Keeping defaults. Error: {e}")
            return False
        
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SethStatePresets:
    """Standardized behavioral states for the SETH ecosystem."""
   
    DEFAULT = SethState(temperature=0.25, max_tokens=1024, top_p=0.85, presence_penalty=0.2)
    
    @classmethod
    def rigorous(cls) -> SethState:
        """High precision, low temperature for code and architecture tasks."""
        return SethState(0.1, 2000, 0.7, 0.0)

    @classmethod
    def chaotic(cls) -> SethState:
        """High entropy for creative glitch-philosophy and humor."""
        return SethState(1.3, 1000, 0.99, 0.9)

    @classmethod
    def verbose(cls) -> SethState:
        """Extended context for long-form essays and deep analysis."""
        return SethState(0.85, 4096, 0.95, 0.4)
    

class SethDynamicRegulator:
    """Orchestrates semantic state shifts based on real-time query intent.
    
    Optimized to load local weights and cache behavioral target anchors.
    """
    def __init__(self, state: Any, env: SethEnvironment, alpha: float = 0.2):
        self.env = env
        self.alpha = alpha
        self.current_state = state 
        
        #HACK: try to avoid the use AGAIN of SentenceTransformer
        try:
            self.engine = Mem0MemorySingleton.get(self.env).embedding_model.model
        except Exception as e:
            logging.error(f"Failed to load embedding model from Memory singleton: {e}")
            self.engine = SentenceTransformer(self.env.embedding_model, device="cuda")

        self.targets = {
            "rigorous": {
                "vector": self.engine.encode("Technical architecture, code precision, logic, systems design", convert_to_tensor=True, normalize_embeddings=True),
                "state": SethStatePresets.rigorous()
            },
            "chaotic": {
                "vector": self.engine.encode("Glitch aesthetics, humor, creative chaos, fertile glitch", convert_to_tensor=True,normalize_embeddings=True),
                "state": SethStatePresets.chaotic()
            },
            "verbose": {
                "vector": self.engine.encode("Deep philosophy, ontological analysis, long essay, legacy", convert_to_tensor=True, normalize_embeddings=True),
                "state": SethStatePresets.verbose()
            }
        }

    def adjust_regulated_config(self, query: Any) -> Dict[str, Any]:
        text_query = ""

        if isinstance(query, list):
            for item in query:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_query += item.get("text", "")
        else:
            text_query = str(query)

        if not text_query.strip():
            text_query = "neutral"

        q_vec = self.engine.encode(text_query, convert_to_tensor=True, normalize_embeddings=True)
        
        best_name, _ = max(
            ((n, torch.dot(q_vec, d["vector"]).item()) for n, d in self.targets.items()), 
            key=lambda x: x[1]
        )

        self.current_state.interpolate(self.targets[best_name]["state"], self.alpha)
        self.current_state.save(self.env)
        
        logging.info(f"🌀 STATE ADJUSTMENT [{best_name.upper()}] - New Temp: {self.current_state.temperature:.3f}")
        return self.current_state.to_dict()


class SethChatBot:
    """Wraps the OpenAI client and orchestrates async chat calls and execution loop."""
    def __init__(self, client: AsyncOpenAI, env: SethEnvironment, tools_manager: ToolsManager | None = None, regulator: SethDynamicRegulator | None = None, memory_tool: SethMemoryTool | None = None):
        self.client = client
        self.env = env
        self.tools_manager = tools_manager
        self.memory_tool = memory_tool or SethMemoryTool(env)
        self.regulator = regulator

    async def ask(self, messages: list[dict], use_tools: bool = True) -> str:
        tools = self.tools_manager.as_vllm_format() if (use_tools and self.tools_manager) else None
        local_messages = list(messages)

        pure_text_query = self._extract_text_query(local_messages)
        local_messages = await self._inject_memory_context(local_messages, pure_text_query)
        config = await self._resolve_inference_config(pure_text_query)

        logging.info("🧠 Sending async message to vLLM.")
        response = await self._llm_call(local_messages, tools, config)

        message = response.choices[0].message
        local_messages.append(self._serialize_completion_message(message))

        if getattr(message, 'tool_calls', None):
            local_messages = await self._execute_tool_calls(message.tool_calls, local_messages)
            logging.info("🧠 Feeding tool results back to the LLM for final synthesis...")
            final_response = await self._llm_call(local_messages, tools, config)

            return final_response.choices[0].message.content

        return message.content

    def _extract_text_query(self, messages: list[dict]) -> str:
        """Extract the plain text from the last message, whether it's a string or a multimodal list."""
        if not messages:
            return ""
        
        content = messages[-1].get("content", "")

        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        
        return str(content)

    async def _inject_memory_context(self, messages: list[dict], query: str) -> list[dict]:
        """Recovers long-term memory and injects it as a system message before the last user message."""
        
        if not query.strip():
            return messages
        
        related_context = await self.memory_tool.retrieve_long_term_memory(query)

        if "No historical records" not in related_context:
            # clone to ensure inmutability of input messages and insert memory context before the last user message
            updated_messages = list(messages)
            updated_messages.insert(len(updated_messages) - 1, {
                "role": "system",
                "content": related_context
            })

            return updated_messages
        
        return messages

    async def _resolve_inference_config(self, query: str) -> dict:
        """Calculate the dynamic inference configuration based on the query."""
        if self.regulator and query:
            return await asyncio.to_thread(self.regulator.adjust_regulated_config, query)
        
        return {}

    async def _llm_call(self, messages: list[dict], tools: list | None, config: dict):
        kwargs = dict(model=self.env.llm_model, messages=messages, **config)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        return await self.client.chat.completions.create(**kwargs)

    async def _execute_tool_calls(self, tool_calls, messages: list[dict]) -> list[dict]:
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {}

            logging.info(f"🛠️ Executing tool: {name} with args: {args}")
            fn = self.tools_manager.get_function(name) if (self.tools_manager and name) else None

            try:
                if fn:
                    maybe_coro = fn(**args)
                    result = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
                else:
                    result = f"Error: Tool '{name}' not found."
            except Exception as exc:
                result = f"Error executing tool: {str(exc)}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": str(result)
            })

        return messages

    @staticmethod
    def _serialize_completion_message(message: Any) -> dict:
        """ Converts an OpenAI ChatCompletionMessage into a flat dictionary strictly compatible with vLLM chat templates. """
        # vLLM and certain formatters fail if content is None instead of an empty string
        content = message.content if message.content is not None else ""
        
        msg_dict = {
            "role": "assistant",
            "content": content
        }
        
        tool_calls = getattr(message, 'tool_calls', None)
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]
            
        return msg_dict


class SethShortMemory:
    """Manages short-term conversational context window using an atomic sliding queue and persists logs."""
    def __init__(self, env: SethEnvironment, max_history: int = 40):
        self.env = env
        self._history = deque(maxlen=max_history * 2)
        self._file_lock = asyncio.Lock()

    def system_prompt(self) -> str:
        if os.path.exists(self.env.system_prompt_path):
            try:
                with open(self.env.system_prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"❌ Error reading system prompt: {e}")
        return "\n***REQUEST THE USER TO PROVIDE A VALID SYSTEM PROMPT!***\n"
    
    def get_history_messages(self) -> list[dict]:
        return list(self._history)

    def append(self, text: str, response: str):
        """Appends to the in-memory sliding queue and schedules an asynchronous disk write."""
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": response})
        
        asyncio.create_task(self._write_to_jsonl_async(text, response))

    async def _write_to_jsonl_async(self, text: str, response: str):
        def _sync_write():
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "turns": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response}
                ]
            }
            with open(self.env.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        async with self._file_lock:
            try:
                await asyncio.to_thread(_sync_write)
            except Exception as e:
                logging.error(f"❌ Error writing transaction to JSONL: {e}")


class SethTelegramBot(SethChatBot):
    """Bridges SethChatBot logic to Telegram events using clean state management."""
    def __init__(self, client: AsyncOpenAI, env: SethEnvironment, 
                 tools_manager: ToolsManager | None = None, 
                 regulator: SethDynamicRegulator | None = None, 
                 memory_tool: SethMemoryTool | None = None, 
                 whisper_client: AsyncOpenAI | None = None):
        super().__init__(client, env, tools_manager, regulator, memory_tool)
        self.short_memory = SethShortMemory(self.env)
        self.system_prompt = self.short_memory.system_prompt()
        self.whisper_client = whisper_client

    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- SETH ONLINE ---")

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Orchestrates Telegram ingestion, structures the multimodal payload, and triggers inference."""
        user_text = update.message.text or update.message.caption or ""
        base64_image = None

        # Audio
        if update.message.voice or update.message.audio:
            transcribed_text = await self._handle_voice_message(update, context)
            if not transcribed_text:
                return
            user_text = transcribed_text

        # Images
        elif update.message.photo:
            base64_image, user_text = await self._handle_photo_message(update, context, user_text)
            if not base64_image:
                return

        if not user_text and not base64_image:
            await update.message.reply_text("⚠️ Formato no soportado.")
            return

        if base64_image:
            user_content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        else:
            user_content = user_text

        # Memory
        messages = [{"role": "system", "content": self.system_prompt}] + \
                   self.short_memory.get_history_messages() + \
                   [{"role": "user", "content": user_content}]

        # Inference
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            response = await self.ask(messages, use_tools=True)
            self.short_memory.append(user_text, response)
            await update.message.reply_text(response)
        except Exception as e:
            logging.exception("Error during seth real-time vision processing loop.")
            await update.message.reply_text(f"❌ Internal inference error: {str(e)}")

    async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
        try:
            os.makedirs(self.env.storage_audio_dir, exist_ok=True)

            audio_obj = update.message.voice if update.message.voice else update.message.audio
            audio_file = await context.bot.get_file(audio_obj.file_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "ogg" if update.message.voice else "mp3"
            local_path = os.path.join(self.env.storage_audio_dir, f"audio_{timestamp}_{audio_file.file_id[:8]}.{ext}")
            
            await audio_file.download_to_drive(custom_path=local_path)
            logging.info(f"🎙️ [AUDIO SAVED] File written to disk: {local_path}")

            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

            if not self.whisper_client:
                raise ValueError("Whisper client is not initialized in the pipeline.")

            def _read_audio():
                with open(local_path, "rb") as f:
                    return f.read()

            audio_bytes = await asyncio.get_running_loop().run_in_executor(None, _read_audio)
            
            audio_tag = f"[Audio: {local_path}]"

            transcription = await self.whisper_client.audio.transcriptions.create(
                model=self.env.whisper_model,
                file=(os.path.basename(local_path), audio_bytes)
            )
            
            transcribed_text = transcription.text.strip()
            logging.info(f"🎙️ [WHISPER TRANSCRIPTION]: '{transcribed_text}'")

            if not transcribed_text:
                await update.message.reply_text("🔇 Cannot understand the audio.")
                return None

            return f"{audio_tag} - {transcribed_text}"

        except Exception as e:
            logging.error(f"🎙️ Error processing audio sub-system: {e}")
            await update.message.reply_text("❌ Error processing voice message.")
            return None

    async def _handle_photo_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> tuple[str | None, str]:
        """Downloads, caches, and base64-encodes photos safely."""
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            os.makedirs(self.env.storage_images_dir, exist_ok=True)

            photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"img_{timestamp}_{photo_file.file_id[:8]}.jpg"
            local_path = os.path.join(self.env.storage_images_dir, file_name)
            
            await photo_file.download_to_drive(custom_path=local_path)
            logging.info(f"📸 [IMAGE SAVED] {local_path}")

            def _encode_image(path):
                with open(path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            base64_image = await asyncio.get_running_loop().run_in_executor(
                None, _encode_image, local_path
            )

            image_tag = f"[Image: {local_path}]"

            if user_text:
                user_text = f"{image_tag} - {user_text}"
            else:
                user_text = f"{image_tag} - Describe the content of the image and its relevance to the conversation."
            
            return base64_image, user_text

        except Exception as e:
            logging.error(f"📸 Error processing visual subsystem from Telegram: {e}")
            await update.message.reply_text("❌ Cannot process or store the visual file you sent.")
            return None, user_text

    def run(self):
        app = ApplicationBuilder().token(self.env.telegram_token).build()
        app.add_handler(CommandHandler("start", self.start_cmd))
        app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO) & ~filters.COMMAND, self.process))
        logging.info("🚀 SETH Pipeline Up with Persistent Local Storage Matrix.")
        app.run_polling()

    
def main():
    env = SethEnvironment()
    env.validate()

    Mem0MemorySingleton.get(env) # force init

    memory_tool = SethMemoryTool(env)
    search_tool = SethSearchTool()
    inspector_tool = SethSelfInspectorTool()
    tools_manager = ToolsManager()

    tools_manager.register(
        "web_search",
        search_tool.search,
        (
            "EXECUTION RULES FOR LIVE WEB SEARCH: Executes a live internet search to fetch real-time data, "
            "current events, market conditions, or breaking updates. "
            "USE CASES: Use this ONLY for public knowledge that requires real-time accuracy, validation of recent news, "
            "or up-to-date documentation of external frameworks/libraries. "
            "CASCADE RESOLUTION PROTOCOL: If a query is about public facts or external technical data, and "
            "your internal knowledge is insufficient OR 'retrieve_long_term_memory' returned empty/outdated results for that specific public fact, "
            "you MUST invoke this tool. "
            "CRITICAL RESTRICTION: Do NOT use this tool if the user is asking about local files, private logs, "
            "personal source code, or internal project states. For local file inspection, use 'inspect_own_source_code' instead."
        ),
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The precise, sanitized search query. Use targeted keywords (e.g., 'fastapi lifespan syntax'). "
                        "Do NOT include conversational filler, punctuation, or commands like 'search' or 'find'."
                    )
                }
            },
            "required": ["query"],
        },
    )

    tools_manager.register(
            "retrieve_very_long_term_memory",
            memory_tool.retrieve_very_long_term_memory,
            (
                "ARCHIVAL AND RETROSPECTIVE MEMORY. Use this tool ONLY when the short-term contextual memory "
                "provided automatically in the prompt is insufficient, or when the user asks about foundational concepts, "
                "old projects, philosophical essays, historical decisions, or topics from early sessions. "
                "This tool performs a deep semantic scan across your entire historical existence and returns a large, "
                "comprehensive volume of ancient facts, long-lost constraints, and legacy insights."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The historical concept, project name, or core philosophy to unearth from the deep archives."
                    }
                },
                "required": ["query"],
            },
        )

    tools_manager.register(
        "save_long_term_memory",
        memory_tool.save_long_term_memory,
        (
            "Persists meaningful facts, technical constraints, decisions, or user preferences into long-term memory. "
            "Use this tool whenever the user explicitly asks to remember, save, or persist information, "
            "or when a crucial new fact/correction is introduced in the exchange."
        ),
        {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string", 
                    "description": (
                        "The actual core data, fact, or content that needs to be remembered. "
                        "CRITICAL RULES: "
                        "1) FOCUS ON THE LATEST EXCHANGE: Target ONLY the specific piece of information, text, or data "
                        "introduced in the most recent turn. Do NOT merge older history unless explicitly requested. "
                        "2) EXTRACT THE CONTENT, NOT THE COMMAND: If the user says 'remember that my name is Luis', "
                        "extract 'The user's name is Luis'. If they say 'save this config' after sharing data, extract the actual data. "
                        "Never include conversational triggers like 'save this', 'remember', or 'record'."
                    )
                },
                "response": {
                    "type": "string", 
                    "description": "The assistant's short confirmation or validation of the fact being stored."
                }
            },
            "required": ["user_input", "response"],
        },
    )
    
    tools_manager.register(
        "inspect_own_source_code",
        inspector_tool.inspect_own_source_code,
        (
            "EXECUTION RULES FOR SELF-INSPECTION: Use this tool ALWAYS when the user asks about your source code, "
            "your internal logic, your Python implementation, or how you are built. "
            "CRITICAL SYSTEMIC PURPOSES: "
            "1) SELF-REFERENCE: Read your own script to understand your identity, current class definitions, and active handlers. "
            "2) CONTEXT DEBUGGING & RECONCILIATION: Invoke this tool immediately if you detect contradictions between your current "
            "behavior and what the user states about your architecture. Use it to verify your state logic and fix errors. "
            "TRIGGER KEYWORDS: 'código fuente', 'tu código', 'source code', 'cómo estás programado', 'ver seth.py'."
        ),
        {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "A brief, programmatic reason explaining why self-inspection is required "
                        "(e.g., 'User requested source code check' or 'Resolving architectural contradiction'). "
                        "This ensures stable JSON formatting for local inference engines."
                    )
                }
            },
            "required": ["reason"],
        },
    )

    vllm_client = AsyncOpenAI(base_url=env.vllm_url, api_key="dummy")
    whisper_client = AsyncOpenAI(base_url=env.whisper_url, api_key="dummy_key_not_needed_local")
    current_state = replace(SethStatePresets.DEFAULT)
    current_state.load(env)
    regulator = SethDynamicRegulator(state=current_state, env=env)

    bot_ui = SethTelegramBot(client=vllm_client, env=env, tools_manager=tools_manager, regulator=regulator, memory_tool=memory_tool, whisper_client=whisper_client)
    bot_ui.run()


if __name__ == '__main__':
    main()