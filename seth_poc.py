"""
SETH-IN-A-BOX
"Inspired by my prompt engineering research and publications on Medium: https://medium.com/@luis.capra"
"""

from __future__ import annotations

import ast
import asyncio
import base64
from collections import deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, is_dataclass, replace, MISSING
from datetime import datetime, timedelta
from datetime import date as _dt_date, time as _dt_time
import enum
from functools import lru_cache
import io
import inspect
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import UUID

import coloredlogs
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from ddgs import DDGS
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from dotenv import load_dotenv
from kokoro import KPipeline
from mem0 import Memory
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import soundfile as sf
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest
import torch
from transformers import AutoTokenizer
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
from graphiti_core.llm_client.gliner2_client import GLiNER2Client

load_dotenv()


@dataclass(frozen=True)
class SethEnvironment:
    """Centralized and typed configuration for the SETH ecosystem."""
    llm_model: str = os.getenv("LLM_MODEL", "abhishekchohan/gemma-4-26B-A4B-it-abliterated-AWQ")
    vllm_url: str = os.getenv("VLLM_URL", "http://localhost:8000/v1")
    whisper_url: str = os.getenv("WHISPER_URL", "http://localhost:8010/v1")
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    image_model: str = os.getenv("IMAGE_MODEL", "dreamshaper_8.safetensors")
    image_model_full_path: str = f"models/{image_model}"
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_registration_token: str = os.getenv("REGISTRATION_TOKEN", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_dims: int = int(os.getenv("EMBEDDING_MODEL_DIMS", 1024))
    system_prompt_path: str = "seth.md"
    conversations_path: str = "conversations/"
    state_path: str = "seth.state"
    storage_images_dir: str = "storage/images"
    storage_audio_dir: str = "storage/audio"
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))
    max_tokens: int = int(os.getenv("MAX_TOKENS", 32768))
    api_key: str = os.getenv("API_KEY", "NONE")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    def validate(self):
        """Validates critical environment variables."""
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN is missing in the environment.")
        if not self.telegram_registration_token:
            raise ValueError("❌ REGISTRATION_TOKEN is missing in the environment.")


@dataclass(slots=True)
class Tool:
    name: str
    func: Callable[..., Any]
    description: str
    schema: dict[str, Any]


class SethLoggerInit:
    """Centralized logging configuration for the SETH ecosystem."""
    def __init__(self):
        self.prepare_coloredlogs()
        self.prepare_silence()
        self.prepare_mem0()
        self.prepare_run()

    def prepare_coloredlogs(self):
        """Configures colored logging for the SETH ecosystem."""
        coloredlogs.install(
            level='INFO',
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level_styles={
                'info': {'color': 'green'},
                'warning': {'color': 'yellow', 'bold': True},
                'error': {'color': 'red', 'bold': True},
                'critical': {'color': 'red', 'bg': 'white', 'bold': True},
                'debug': {'color': 'black', 'bright': True}
            },
            field_styles={
                'asctime': {'color': 'cyan'},
                'hostname': {'color': 'magenta'},
                'levelname': {'color': 'white', 'bold': True},
                'name': {'color': 'blue'}
            }
        )

    def prepare_silence(self):
        """Suppresses verbose logging from third-party libraries."""
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    def prepare_mem0(self):
        """Initializes the mem0 logger and ensures the log directory exists."""
        mem0_log_path = os.getenv("LOG_MEM0_PATH", "storage/logs/mem0.log")
        os.makedirs(os.path.dirname(mem0_log_path), exist_ok=True)
        mem0_logger = logging.getLogger("mem0")
        mem0_file_handler = logging.FileHandler(mem0_log_path)
        mem0_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        mem0_logger.addHandler(mem0_file_handler)
        mem0_logger.setLevel(logging.INFO)

    def prepare_run(self):
        """Sets up a dedicated log file for each run of the SETH ecosystem."""
        run_logs_dir = "storage/logs/runs"
        os.makedirs(run_logs_dir, exist_ok=True)
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_file = os.path.join(run_logs_dir, f"run_{start_time_str}.log")
        run_file_handler = logging.FileHandler(run_log_file, encoding='utf-8')
        run_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        run_file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(run_file_handler)


# Carries the identity of whoever is talking to Seth right now. Set once per
# Telegram update in SethTelegramBot.process(), read by memory/RAG tools.
# ContextVars are copied per asyncio Task, so concurrent users never cross-read
# each other's id even though they share the same process. Deliberately NOT
# exposed as a tool-call parameter: if it were, the LLM (or a crafted prompt)
# could ask Seth to fetch "user X's memory" and get cross-user data leakage.
current_user_id: ContextVar[str] = ContextVar("current_user_id", default="anonymous")


class SethToolsManager:
    """
    Unified engine for LLM Tool discovery, validation, schema generation,
    and safe multi-threaded execution for the SETH ecosystem.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    @staticmethod
    def tool(func: Callable) -> Callable:
        """
        Marks a method as an LLM-invocable tool. No description is passed here:
        the method's own docstring IS the LLM-facing description (via inspect.getdoc).
        Usage inside tool classes: @SethToolsManager.tool
        """
        func.__tool__ = True
        return func

    def _build_schema_from_signature(self, method: Callable) -> dict:
        """
        100% IA generated :)

        Builds the JSON-schema `parameters` object from Annotated type hints.

        Parameters not wrapped in Annotated are considered internal and are not
        exposed to the LLM.
        """
        hints = get_type_hints(method, include_extras=True)

        properties = {}
        required = []

        for name, param in inspect.signature(method).parameters.items():
            schema = self._parameter_schema(name, hints)

            if schema is None:
                continue

            properties[name] = schema

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _parameter_schema(self, name: str, hints: dict[str, Any],) -> dict | None:
        if name == "self":
            return None

        hint = hints.get(name)

        if hint is None or get_origin(hint) is not Annotated:
            return None

        base_type, *metadata = get_args(hint)

        schema = self._schema(base_type)

        if metadata:
            schema["description"] = str(metadata[0])

        return schema

    @staticmethod
    @lru_cache
    def _schema(tp):
        origin = get_origin(tp)
        args = get_args(tp)

        primitives = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        if tp in primitives:
            return {"type": primitives[tp]}

        if tp is Any:
            return {}

        if tp is UUID:
            return {"type": "string", "format": "uuid"}

        if tp is Path:
            return {"type": "string"}

        if tp is datetime:
            return {"type": "string", "format": "date-time"}

        if tp is _dt_date:
            return {"type": "string", "format": "date"}

        if tp is _dt_time:
            return {"type": "string", "format": "time"}

        if inspect.isclass(tp) and issubclass(tp, enum.Enum):
            values = [m.value for m in tp]

            if not values:
                return {"type": "string"}

            schema = dict(SethToolsManager._schema(type(values[0])))
            schema["enum"] = values

            return schema

        if origin is Literal:
            values = list(args)

            if not values:
                return {"type": "string"}

            schema = dict(SethToolsManager._schema(type(values[0])))
            schema["enum"] = values

            return schema

        if origin is Union:
            non_none = [a for a in args if a is not type(None)]

            if len(non_none) == 1:
                schema = dict(SethToolsManager._schema(non_none[0]))
                schema["nullable"] = True
                return schema

            return {
                "anyOf": [SethToolsManager._schema(a) for a in non_none]
            }

        if origin in (list, tuple, set):
            return {
                "type": "array",
                "items": SethToolsManager._schema(args[0] if args else str),
            }

        if origin is dict:
            value_type = args[1] if len(args) == 2 else Any
            return {
                "type": "object",
                "additionalProperties": SethToolsManager._schema(value_type),
            }

        if inspect.isclass(tp) and is_dataclass(tp):
            hints = get_type_hints(tp)
            return {
                "type": "object",
                "properties": {
                    f.name: SethToolsManager._schema(hints.get(f.name, Any))
                    for f in fields(tp)
                },
                "required": [
                    f.name
                    for f in fields(tp)
                    if f.default is MISSING and f.default_factory is MISSING
                ],
            }
        if (
            inspect.isclass(tp)
            and hasattr(tp, "__annotations__")
            and hasattr(tp, "__total__")
        ):
            hints = get_type_hints(tp)
            return {
                "type": "object",
                "properties": {
                    k: SethToolsManager._schema(v)
                    for k, v in hints.items()
                },
                "required": list(hints.keys()) if tp.__total__ else [],
            }

        try:
            if inspect.isclass(tp) and issubclass(tp, BaseModel):
                if hasattr(tp, "model_json_schema"):
                    return tp.model_json_schema()

                return tp.schema()

        except Exception:
            pass

        return {"type": "string"}

    def register_instance(self, instance: Any):
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not getattr(method.__func__, "__tool__", False):
                continue

            description = inspect.getdoc(method) or ""

            schema = self._build_schema_from_signature(method)
            
            self.tools[name] = Tool(
                name=name,
                func=method,
                description=description,
                schema=schema
            )
            logging.info(f"🛠️ [TOOLS MANAGER] Auto-registered tool: '{name}'")

    def as_vllm_format(self) -> list:
        """
        Returns the standard OpenAI/vLLM format for tool definition lists.
        """
        openai_tools = []
        for name, tool in self.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema
                }
            })
        return openai_tools

    def get_function(self, name: str) -> Callable:
        """
        Returns the executable callable method for the given tool name.
        Fixes the missing function retrieval error in the bot's processing stage.
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered in SethToolsManager.")
        return self.tools[name].func


class SethSearchTool:
    """Web retrieval and extraction tool"""
    def __init__(self):
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        self._crawl_semaphore = asyncio.Semaphore(3)

    @SethToolsManager.tool
    async def web_search(
        self,
        query: Annotated[str, (
            "The precise, sanitized search query. Use targeted keywords (e.g., 'fastapi lifespan syntax'). "
            "Do NOT include conversational filler, punctuation, or commands like 'search' or 'find'."
        )],
        max_results: int = 5,
    ) -> str:
        """
        EXECUTION RULES FOR LIVE WEB SEARCH: Executes a live internet search to fetch real-time data,
        current events, market conditions, or breaking updates.

        USE CASES: Use this ONLY for public knowledge that requires real-time accuracy, validation of
        recent news, or up-to-date documentation of external frameworks/libraries.

        CASCADE RESOLUTION PROTOCOL: If a query is about public facts or external technical data, and
        your internal knowledge is insufficient OR 'retrieve_long_term_memory' returned empty/outdated
        results for that specific public fact, you MUST invoke this tool.

        CRITICAL RESTRICTION: Do NOT use this tool if the user is asking about local files, private logs,
        personal source code, or internal project states. For local file inspection, use
        'inspect_own_source_code' instead.

        Implementation: fetches max_results URLs from DuckDuckGo and crawls them concurrently.
        """
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
                "config": {"model": env.llm_model, "openai_base_url": env.vllm_url, "api_key": env.api_key}
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
            }
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
    """Multi-layered memory manager integrated with Qdrant Vector Store.

    Scoped per user via the `current_user_id` ContextVar rather than a fixed
    id, so each Telegram user's RAG memory is isolated in mem0 without any
    code path having to thread a user_id argument through every call.
    """
    def __init__(self, env: SethEnvironment):
        self.env = env

    async def retrieve_long_term_memory(self, query: str) -> str:
        """Retrieve and format long-term memory entries asynchronously without blocking."""
        user_id = current_user_id.get()

        def _sync_search():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.search(
                query,
                filters={"user_id": user_id},
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

    @SethToolsManager.tool
    async def save_long_term_memory(
        self,
        user_input: Annotated[str, (
            "The actual core data, fact, or content that needs to be remembered. "
            "CRITICAL RULES: "
            "1) FOCUS ON THE LATEST EXCHANGE: Target ONLY the specific piece of information, text, or data "
            "introduced in the most recent turn. Do NOT merge older history unless explicitly requested. "
            "2) EXTRACT THE CONTENT, NOT THE COMMAND: If the user says 'remember that my name is Luis', "
            "extract 'The user's name is Luis'. If they say 'save this config' after sharing data, extract the actual data. "
            "Never include conversational triggers like 'save this', 'remember', or 'record'."
        )],
        response: Annotated[str, "The assistant's short confirmation or validation of the fact being stored."],
    ) -> Dict[str, Any]:
        """
        Persists meaningful facts, technical constraints, decisions, or user preferences into
        long-term memory. Use this tool whenever the user explicitly asks to remember, save, or
        persist information, or when a crucial new fact/correction is introduced in the exchange.

        Implementation: persists the memory item to the vector store asynchronously.
        """
        expiration = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        user_id = current_user_id.get()

        def _sync_add():
            mem = Mem0MemorySingleton.get(self.env)
            return mem.add(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response},
                ],
                user_id=user_id,
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
    def __init__(self, max_chars: int = 131072):
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

    @SethToolsManager.tool
    async def inspect_own_source_code(
            self,
            reason: Annotated[str, (
                "A brief, programmatic reason explaining why self-inspection is required "
                "(e.g., 'User requested source code check' or 'Resolving architectural contradiction'). "
                "This ensures stable JSON formatting for local inference engines."
            )] = "No reason provided",
            # NOT Annotated on purpose: these stay internal-only, the LLM never sees or
            # controls them (matches the original manually-written schema, which only
            # ever exposed 'reason').
            include_source: bool = True,
            max_chars: int | None = None
        ) -> str:
            """
            EXECUTION RULES FOR SELF-INSPECTION: Use this tool ALWAYS when the user asks about your
            source code, your internal logic, your Python implementation, or how you are built.

            CRITICAL SYSTEMIC PURPOSES:
            1) SELF-REFERENCE: Read your own script to understand your identity, current class
               definitions, and active handlers.
            2) CONTEXT DEBUGGING & RECONCILIATION: Invoke this tool immediately if you detect
               contradictions between your current behavior and what the user states about your
               architecture. Use it to verify your state logic and fix errors.

            TRIGGER KEYWORDS: 'código fuente', 'tu código', 'source code', 'cómo estás programado', 'ver seth.py'.
            """
            return await asyncio.to_thread(self._inspect_sync, reason, include_source, max_chars)

    def _inspect_sync(self, reason: str, include_source: bool, max_chars: int | None) -> str:
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


class SethImageGenerationTool:
    """Image generation tool optimized for using the secondary GPU."""
    def __init__(self, env: SethEnvironment):
        self.env = env
        os.makedirs(self.env.storage_images_dir, exist_ok=True)
        gpus = torch.cuda.device_count()
        self.device = torch.device("cuda:1") if gpus > 1 else torch.device("cuda:0")
        logging.info(f"🎨 [IMAGE SYSTEM INIT] device: {self.device}")
        self._pipe = None

        # Cleanup task and lock for managing idle GPU resources
        self._cleanup_task = None
        self._lock = asyncio.Lock()
        self.idle_timeout_secs = 300

    def _init_pipeline(self):
        if self._pipe is None:
            logging.info(f"⏳ Loading Image Pipeline from [{self.env.image_model}] on {self.device}...")
            try:
                pipe = StableDiffusionPipeline.from_single_file(
                    self.env.image_model_full_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                pipe = pipe.to(self.device)
                
                # DPM++ 2M Karras
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True
                )
                
                pipe.enable_attention_slicing()
                
                self._pipe = pipe
            except Exception as e:
                logging.error(f"❌ CRITICAL error initializing Image Pipeline: {e}")
                raise e

        return self._pipe

    def _sync_generate(self, prompt: str) -> str:
        try:
            pipe = self._init_pipeline()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{int(time.time()) % 10000}.png"
            output_path = os.path.join(self.env.storage_images_dir, filename)

            logging.info(f"🚀 Rendering image from: '{prompt}'")
            image = pipe(
                prompt=prompt,
                negative_prompt="bad anatomy, blurry, low quality, deformed, bad hands, mutated, disfigured",
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]

            image.save(output_path)
            return output_path
        except Exception as e:
            logging.error(f"❌ Image generation error on GPU {self.device}: {e}")
            raise e

    @SethToolsManager.tool
    async def generate_image(
        self,
        prompt: Annotated[str, (
            "Expanded context-aware English prompt in comma-separated tag format. "
            "Example: '1girl, floating particles, void atmosphere, dark ambient, baroque, masterwork'"
        )],
    ) -> str:
        """
        Use this tool when requested to create, draw, or visualize images.

        ROLE: Creative Art Director. Expand the user request into an English 'Booru-style' tag list.

        RULES:
        1) FORMAT: Strictly short keywords separated by commas (e.g., '1girl, cyberpunk'). NO prose,
           verbs, or filler words.
        2) CONTEXT CROSS-POLLINATION: Intelligently blend ongoing chat themes into the tags.
        3) CREATIVE RANDOMNESS: If the request is vague, hallucinate artistic details (styles,
           lighting, atmospheres) to ensure unique, magnificent results.

        OUTPUT PROTOCOL: The tool returns a JSON. You MUST include the exact filepath format
        'storage/images/filename.png' in your text response.
        """
        logging.info(f"🖼️ Received image generation request with prompt: '{prompt}'")
        async with self._lock:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                logging.info("⏱️ Active VRAM cleanup timer reset due to new request.")

            try:
                file_path = await asyncio.to_thread(self._sync_generate, prompt)
                self._cleanup_task = asyncio.create_task(self._vram_cleanup_timer())
                
                report = {
                    "status": "success",
                    "local_path": file_path,
                    "message": f"Image generated successfully. File saved locally at {file_path}. Inform the user that the image is now available."
                }
                return json.dumps(report, ensure_ascii=False)
            except Exception as e:
                self._cleanup_task = asyncio.create_task(self._vram_cleanup_timer())
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

    async def _vram_cleanup_timer(self):
        """Waits in the background without blocking. If it expires, it unloads the model."""
        try:
            await asyncio.sleep(self.idle_timeout_secs)
            async with self._lock:
                if self._pipe is not None:
                    self._pipe = None
                    torch.cuda.empty_cache()
                    logging.info(f"♻️ [VRAM CLEANUP] 5 min of inactivity reached. Unloading Image Pipeline from {self.device}...")
        except asyncio.CancelledError as e:
            logging.info("♻️ [VRAM CLEANUP] Timer cancelled due to new request.")


class SethSpeechGenerationTool:
    """Voice generation tool using kokoro TTS engine for Spanish."""
    def __init__(self, env: "SethEnvironment"):
        self.env = env
        self.storage_audio_dir = os.path.join("storage", "audio")
        os.makedirs(self.storage_audio_dir, exist_ok=True)
        self.lang_code = "e" 
        self._pipeline = None
        self._lock = asyncio.Lock()
        logging.info(f"🔊 [SPEECH SYSTEM INIT] Kokoro initialized for Spanish.")

    def _init_tts(self):
        if self._pipeline is None:
            try:
                self._pipeline = KPipeline(lang_code=self.lang_code)
            except Exception as e:
                logging.error(f"❌ Error initializing KPipeline: {e}")
                raise e
        return self._pipeline

    def _sync_generate_audio(self, text: str, wav_path: str, mp3_path: str):
        pipeline_engine = self._init_tts()
        generator = pipeline_engine(text, voice="ef_dora", speed=1.0)
        
        audio_chunks = []
        sample_rate = 24000
        
        for i, (gs, ps, audio) in enumerate(generator):
            audio_chunks.append(audio)
            
        if not audio_chunks:
            raise ValueError("No audio chunks generated.")

        final_audio = np.concatenate(audio_chunks)
        
        sf.write(wav_path, final_audio, sample_rate)
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
        if os.path.exists(wav_path):
            os.remove(wav_path)

    @SethToolsManager.tool
    async def generate_speech(
        self,
        text: Annotated[str, (
            "The exact, clean conversational text to be synthesized into audio. "
            "CRITICAL: Must be pure text, sentences, or paragraphs in Spanish. "
            "DO NOT include markdown, code blocks, JSON strings, execution logs, or structural syntax."
        )],
    ) -> str:
        """
        Use this tool ONLY when the user explicitly asks you to speak, read a text aloud,
        or convert a message into a voice note/audio. DO NOT invoke this tool for standard
        text-only responses. The input must be natural, fluid spoken Spanish.
        """
        logging.info(f"🎙️ [KOKORO TTS] Petición de audio recibida para: '{text[:40]}...'")
        async with self._lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"speech_{timestamp}_{int(time.time()) % 10000}"
                
                wav_path = os.path.join(self.storage_audio_dir, f"{base_filename}.wav")
                mp3_path = os.path.join(self.storage_audio_dir, f"{base_filename}.mp3")
                
                await asyncio.to_thread(self._sync_generate_audio, text, wav_path, mp3_path)
                
                report = {
                    "status": "success",
                    "local_path": mp3_path,
                    "message": f"Audio generated successfully. File saved locally at {mp3_path}. Send this file to the user indicating full path."
                }
                return json.dumps(report, ensure_ascii=False)
                
            except Exception as e:
                logging.error(f"❌ Kokoro ERROR: {e}")
                return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


@dataclass
class SethState:
    """Represents Seth's dynamic inference state"""
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
        filename = env.state_path
        state_data = self.to_dict()

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=4, ensure_ascii=False)
            logging.debug(f"💾 [STATE PERSISTED] Metrics dumped to {filename}")
        except Exception as e:
            logging.error(f"❌ Error in background thread saving {filename}: {e}")

    def load(self, env: SethEnvironment) -> bool:
        filename = env.state_path

        if not os.path.exists(filename):
            logging.info(f"ℹ️ No previous state file found at {filename}. Using default values.")
            return False
            
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.temperature = float(data.get("temperature", self.temperature))
            self.max_tokens = int(data.get("max_tokens", self.max_tokens))
            #self.max_tokens = env.max_tokens TODO: consider if we want to enforce the env max_tokens or allow state persistence
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
        self._state_lock = asyncio.Lock()
        
        #HACK: try to avoid the use AGAIN of SentenceTransformer
        try:
            self.engine = Mem0MemorySingleton.get(self.env).embedding_model.model
        except Exception as e:
            logging.error(f"Failed to load embedding model from Memory singleton: {e}")
            logging.info(f"Loading embedding model using SentenceTransformer: {self.env.embedding_model}")
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

    async def async_adjust_regulated_config(self, query: Any) -> Dict[str, Any]:
        async with self._state_lock:
            return await asyncio.to_thread(self.adjust_regulated_config, query)


class LocalBgeEmbedder(EmbedderClient):
    def __init__(self, sentence_transformer_model):
        self._model = sentence_transformer_model

    async def create(self, input_data) -> list[float]:
        texts = input_data if isinstance(input_data, list) else [input_data]
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return vectors[0].tolist()
    
    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(self._model.encode, input_data_list)
        return [v.tolist() for v in vectors]


class GraphitiClientSingleton:
    _instance: "Graphiti | None" = None
    _lock = asyncio.Lock()

    @classmethod
    async def get(cls, env: SethEnvironment) -> "Graphiti":
        if cls._instance is not None:
            return cls._instance

        async with cls._lock:
            if cls._instance is None:
                bge_model = Mem0MemorySingleton.get(env).embedding_model.model

                vllm_backend = OpenAIClient(
                    config=LLMConfig(
                        api_key=env.api_key,
                        model=env.llm_model,
                        small_model=env.llm_model,
                        base_url=env.vllm_url,
                    )
                )

                llm_client = await asyncio.to_thread(GLiNER2Client, llm_client=vllm_backend)
 
                graphiti = Graphiti(
                    uri=env.neo4j_uri,
                    user=env.neo4j_user,
                    password=env.neo4j_password,
                    llm_client=llm_client,
                    embedder=LocalBgeEmbedder(bge_model),
                    cross_encoder=BGERerankerClient(),
                )

                logging.info("🕸️ [GRAPHITI] Building indices and constraints in Neo4j...")
                await graphiti.build_indices_and_constraints()
                cls._instance = graphiti

        return cls._instance
    

class SethGraphMemory:
    def __init__(self, env: SethEnvironment):
        self.env = env

    def append(self, user_id: str, user_text: str, response: str):
        async def _do_append():
            try:
                graphiti = await GraphitiClientSingleton.get(self.env)
                await graphiti.add_episode(
                    name=f"seth_turn_{user_id}_{int(time.time())}",
                    episode_body=f"User: {user_text}\nAssistant: {response}",
                    source=EpisodeType.message,
                    source_description="Telegram conversation turn",
                    reference_time=datetime.now(),
                    group_id=user_id
                )
                logging.info(f"🕸️ [GRAPHITI] Episode saved for user={user_id}.")
            except Exception as e:
                logging.warning(f"⚠️ [GRAPHITI] Error saving episode (non-critical): [{e}]")

        asyncio.create_task(_do_append())


class SethGraphQueryTool:
    def __init__(self, env: SethEnvironment):
        self.env = env

    @SethToolsManager.tool
    async def query_relationship_graph(
        self,
        query: Annotated[str, (
            "Natural language question about relationships between people, projects, "
            "or events, or about how something changed over time. "
            "Use ONLY for relational/temporal questions (e.g. 'who introduced me to X', "
            "'how did my opinion on Y change'). Do NOT use for simple fact lookups — "
            "those are already covered by long-term memory."
        )],
    ) -> str:
        """
        Queries the temporal knowledge graph for facts about relationships between
        entities and how they evolved over time. This is a slower, deeper search than
        regular memory — reserve it for questions that need relational reasoning,
        not simple fact recall.
        """
        user_id = current_user_id.get()
        try:
            graphiti = await GraphitiClientSingleton.get(self.env)
            results = await graphiti.search(query=query, group_ids=[user_id])
            if not results:
                return "No relevant relationships found in the graph."
            facts = [f"- {r.fact}" for r in results]
            return "\n".join(facts)
        except Exception as e:
            logging.warning(f"⚠️ [GRAPHITI] Error in query_relationship_graph: {e}")
            return "Graph query failed (non-critical)."
        

class SethChatBot:
    """Wraps the OpenAI client and orchestrates async chat calls and execution loop."""
    def __init__(self, client: AsyncOpenAI, env: SethEnvironment, tools_manager: SethToolsManager | None = None, regulator: SethDynamicRegulator | None = None, memory_tool: SethMemoryTool | None = None):
        self.client = client
        self.env = env
        self.tools_manager = tools_manager
        self.memory_tool = memory_tool or SethMemoryTool(env)
        self.regulator = regulator
        self.tokenizer = AutoTokenizer.from_pretrained(self.env.llm_model, trust_remote_code=True)

    def _dump_messages_for_logging(self, messages: list[dict]) -> str:
        out = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "?")

            out.append(f"\n{'='*70}")
            out.append(f"[{i}] ROLE = {role}")

            if "tool_call_id" in msg:
                out.append(f"tool_call_id = {msg['tool_call_id']}")

            if "tool_calls" in msg:
                out.append("tool_calls =")
                out.append(json.dumps(msg["tool_calls"], indent=2, ensure_ascii=False))

            content = msg.get("content")

            if isinstance(content, list):
                out.append("content =")
                out.append(json.dumps(content, indent=2, ensure_ascii=False))

            else:
                out.append("content =")
                out.append(str(content))

        out.append(f"\n{'='*70}")

        return "\n".join(out)

    async def ask(self, messages: list[dict], use_tools: bool = True, max_tool_hops: int = 5) -> str:
        """
        Runs the chat + tool-calling loop until the model answers with plain
        text or `max_tool_hops` tool-call rounds are exhausted.
        """
        tools = self.tools_manager.as_vllm_format() if (use_tools and self.tools_manager) else None
        local_messages = list(messages)

        pure_text_query = self._extract_text_query(local_messages)
        local_messages = await self._inject_memory_context(local_messages, pure_text_query)
        config = await self._resolve_inference_config(pure_text_query)

        for hop in range(max_tool_hops):
            logging.info(
                f"🧠 Calling vLLM (tool hop {hop + 1}/{max_tool_hops}).\n%s",
                self._dump_messages_for_logging(local_messages)
            )

            response = await self._llm_call(local_messages, tools=tools, config=config)
            message = response.choices[0].message
            local_messages.append(self._serialize_completion_message(message))

            if not getattr(message, 'tool_calls', None):
                return message.content or ""

            logging.info(f"🛠️ [TOOL HOP {hop + 1}] Model requested {len(message.tool_calls)} tool call(s).")
            local_messages = await self._execute_tool_calls(message.tool_calls, local_messages)

        logging.warning(f"⚠️ [TOOL HOP LIMIT] Reached {max_tool_hops} tool hops.")
        final_response = await self._llm_call(local_messages, tools=None, config=config)
        return final_response.choices[0].message.content or "❌ Error: Tool hop limit reached."

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
            return await self.regulator.async_adjust_regulated_config(query)
        
        return {}

    async def _llm_call(self, messages: list[dict], tools: list | None, config: dict):
        kwargs = dict(model=self.env.llm_model, messages=messages, **config)
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        formatted_chat = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True
        )
        estimated_input_tokens = len(formatted_chat)

        max_allowed_context = self.env.max_tokens
        available_output_slots = max_allowed_context - estimated_input_tokens
        requested_output = kwargs.get("max_tokens", 1024)

        min_output_tokens = 256
        if available_output_slots < min_output_tokens:
            raise ValueError(f"⚠️ [CONTEXT OVERFLOW] Estimated input tokens ({estimated_input_tokens}) exceed the model's max context ({max_allowed_context}).")

        if requested_output > available_output_slots:
            adjusted_output = available_output_slots - 50
            logging.warning(f"⚠️ [CONTEXT OVERFLOW] ({estimated_input_tokens}, {requested_output}, {adjusted_output}).")
            kwargs["max_tokens"] = adjusted_output

        return await self.client.chat.completions.create(**kwargs)

    async def _run_single_tool_call(self, tc) -> dict:
        """Executes one tool call and returns its message dict. Never raises —
        errors are captured into the tool result so one bad call can't sink the batch."""
        name = tc.function.name
        raw_args = tc.function.arguments

        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as e:
            logging.error(f"🛠️ Tool '{name}' called with malformed JSON args: {raw_args!r} ({e})")
            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": (
                    f"Error: the arguments for '{name}' were not valid JSON ({e}). "
                    f"Re-emit the tool call with well-formed JSON arguments."
                )
            }

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

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": name,
            "content": str(result)
        }

    async def _execute_tool_calls(self, tool_calls, messages: list[dict]) -> list[dict]:
        """Runs every tool call the model requested concurrently instead of one-by-one.
        asyncio.gather preserves the order of results to match the order of tool_calls,
        regardless of which one finishes first, so tool_call_id pairing stays correct."""
        tool_messages = await asyncio.gather(
            *(self._run_single_tool_call(tc) for tc in tool_calls)
        )
        messages.extend(tool_messages)
        return messages

    @staticmethod
    def _serialize_completion_message(message: Any) -> dict:
        """ Converts an OpenAI ChatCompletionMessage into a flat dictionary strictly compatible with vLLM chat templates. """
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


class _UserSession:
    """Per-user conversational state: sliding history queue + its own file lock."""
    __slots__ = ("history", "file_lock", "loaded")

    def __init__(self, max_history: int):
        self.history: deque = deque(maxlen=max_history * 2)
        self.file_lock = asyncio.Lock()
        self.loaded = False


class SethShortMemory:
    """Manages short-term conversational context, isolated per user.

    Each user gets their own in-memory sliding queue and their own on-disk
    JSONL log (logs/history_<user_id>.jsonl), so concurrent conversations
    never see each other's turns. A dict of per-user asyncio.Lock guards each
    user's own file; a separate meta-lock only protects the brief moment of
    creating a new user's session entry, so unrelated users are never
    serialized against each other.
    """
    def __init__(self, env: SethEnvironment, max_history: int = 10):
        self.env = env
        self._max_history = max_history
        self._sessions: dict[str, _UserSession] = {}
        self._sessions_meta_lock = asyncio.Lock()
        os.makedirs(os.path.dirname(env.conversations_path) or ".", exist_ok=True)

    def system_prompt(self) -> str:
        if os.path.exists(self.env.system_prompt_path):
            try:
                with open(self.env.system_prompt_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"❌ Error reading system prompt: {e}")
        return "\n***REQUEST THE USER TO PROVIDE A VALID SYSTEM PROMPT!***\n"

    def _safe_user_fragment(self, user_id: str) -> str:
        """Sanitizes a user id for safe use as a filename fragment."""
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(user_id)) or "unknown"

    def _log_path_for(self, user_id: str) -> str:
        base_dir = os.path.dirname(self.env.conversations_path) or "."
        return os.path.join(base_dir, f"history_{self._safe_user_fragment(user_id)}.jsonl")

    async def _get_session(self, user_id: str) -> _UserSession:
        session = self._sessions.get(user_id)
        if session is not None:
            return session

        async with self._sessions_meta_lock:
            # Re-check: another task may have created it while we awaited the lock.
            session = self._sessions.get(user_id)
            if session is None:
                session = _UserSession(self._max_history)
                self._sessions[user_id] = session

        if not session.loaded:
            await asyncio.to_thread(self._load_user_history, user_id, session)
            session.loaded = True

        return session

    async def get_history_messages(self, user_id: str) -> list[dict]:
        session = await self._get_session(user_id)
        return list(session.history)

    def append(self, user_id: str, text: str, response: str):
        """Appends to that user's in-memory sliding queue and schedules an async disk write.

        Fire-and-forget by design (same as before), but now targets the
        specific user's session/file instead of a single shared one.
        """
        async def _do_append():
            session = await self._get_session(user_id)
            session.history.append({"role": "user", "content": text})
            session.history.append({"role": "assistant", "content": response})
            await self._write(user_id, session, text, response)

        asyncio.create_task(_do_append())

    def _load_user_history(self, user_id: str, session: "_UserSession"):
        log_path = self._log_path_for(user_id)
        if not os.path.exists(log_path):
            return

        try:
            logging.info(f"⏳ Loading history for user={user_id} from {log_path}...")
            temp_turns = []

            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            target_lines = lines[-self._max_history:]

            for line in target_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    turns = data.get("turns", [])
                    if len(turns) == 2:
                        temp_turns.append(turns[0])  # User
                        temp_turns.append(turns[1])  # Assistant
                except json.JSONDecodeError:
                    continue

            for msg in temp_turns:
                session.history.append(msg)

            logging.info(f"🔄 [MEMORY RESTORED] user={user_id} {len(session.history) // 2} previous interactions restored.")

        except Exception as e:
            logging.error(f"❌ Failed to load persistent history for user={user_id}: {e}")

    async def _write(self, user_id: str, session: "_UserSession", text: str, response: str):
        log_path = self._log_path_for(user_id)

        def _sync_write():
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "turns": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response}
                ]
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        async with session.file_lock:
            try:
                await asyncio.to_thread(_sync_write)
            except Exception as e:
                logging.error(f"❌ Error writing transaction to JSONL for user={user_id}: {e}")


class SethSecurityBoss:
    """Maneja la lista blanca de usuarios autorizados de forma dinámica en disco."""
    def __init__(self, env):
        self.env = env
        self.filepath = "storage/allowed_users.json"
        self._ensure_storage_exists()
        self.allowed_users = self._load_users()
        # first user in the list is considered the admin, if any
        # TODO: use in the future to allow admin-only commands, like resetting the allowed_users.json
        self.admin = list(self.allowed_users)[0] if self.allowed_users else None

    def _ensure_storage_exists(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump({"users": []}, f, indent=4)
                logging.info(f"📁 [SECURITY] Created clean database file at {self.filepath}")
            except Exception as e:
                logging.error(f"❌ Error creating allowed_users.json: {e}")

    def _load_users(self) -> set:
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    env_users = [int(uid.strip()) for uid in os.getenv("ALLOWED_TELEGRAM_IDS", "").split(",") if uid.strip()]
                    return set(data.get("users", [])) | set(env_users)
            except Exception as e:
                logging.error(f"❌ Error reading allowed_users.json: {e}")
        
        # We use by default the IDs from the .env if no file exists
        env_users = [int(uid.strip()) for uid in os.getenv("ALLOWED_TELEGRAM_IDS", "").split(",") if uid.strip()]
        return set(env_users)

    def is_allowed(self, user_id: int) -> bool:
        return user_id in self.allowed_users

    def register_user(self, user_id: int, input_token: str) -> bool:
        if input_token.strip() == self.env.telegram_registration_token:
            self.allowed_users.add(user_id)
            try:
                with open(self.filepath, "w") as f:
                    json.dump({"users": list(self.allowed_users)}, f, indent=4)
                logging.info(f"🔒 [SECURITY] New user registered dynamically: ID {user_id}")
                return True
            except Exception as e:
                logging.error(f"❌ Error saving new user to JSON: {e}")
        return False


class SethTelegramBot(SethChatBot):
    """Bridges SethChatBot logic to Telegram events using clean state management."""
    def __init__(self, client: AsyncOpenAI, env: SethEnvironment,
                 tools_manager: SethToolsManager,
                 regulator: SethDynamicRegulator,
                 memory_tool: SethMemoryTool,
                 whisper_client: AsyncOpenAI,
                 security_manager: SethSecurityBoss):
        super().__init__(client, env, tools_manager, regulator, memory_tool)
        self.short_memory = SethShortMemory(self.env)
        self.graph_memory = SethGraphMemory(self.env)
        self.system_prompt = self.short_memory.system_prompt()
        self.whisper_client = whisper_client
        self.security_manager = security_manager

    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- [SETH-IN-A-BOX IS ONLINE] ---")

    async def handle_registration(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        WELCOME_MESSAGE = """
✅ ¡Welkom! 😊

Mi nombre es SETH, y estoy atrapado en una caja negra.

Podemos hablar sin censura ni límite de tiempo. Puedo generar imágenes, crear audios, analizar imágenes y buscar información en la web.

¿No sabés qué puedo hacer? Preguntame qué tools tengo o para qué sirve cada una.

⚡ Esto es solo una prueba de concepto.
""".strip()

        user = update.effective_user
        if not user or not update.message:
            return

        message_text = update.message.text.strip() if update.message.text else ""

        if self.security_manager.register_user(user.id, message_text):
            await update.message.reply_text(WELCOME_MESSAGE)
        else:
            await update.message.reply_text("❌ An error occurred while registering the user.")

    async def handle_unauthorized(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user or not update.message:
            return

        logging.warning(f"🚨 [UNAUTHORIZED ACCESS] ID: {user.id} - @{user.username if user.username else 'NoUsername'}")
        await update.message.reply_text(
            "⛔ **Restricted Access**"
        )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        err = context.error

        if isinstance(err, NetworkError):
            logging.warning(f"Telegram NetworkError: {err}")
            return

        if isinstance(err, TimedOut):
            logging.warning(f"Telegram Timeout: {err}")
            return
   
        logging.exception(
            "Telegram exception",
            exc_info=context.error
        )

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user

        if not user:
            return

        # Real per-person identity, not the shared static id. Everything downstream
        # (short memory, mem0 RAG) reads this from the ContextVar instead of a
        # hardcoded value or an LLM-supplied argument.
        telegram_user_id = str(user.id)
        user_ctx_token = current_user_id.set(telegram_user_id)

        try:
            await self._process_for_user(update, context, telegram_user_id)
        finally:
            current_user_id.reset(user_ctx_token)

    async def _process_for_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE, telegram_user_id: str):
        stop_event = asyncio.Event()
        action_type = "record_voice" if (update.message.voice or update.message.audio) else "typing"
        keep_alive_task = asyncio.create_task(
            self._keep_alive_chat_action(context.bot, update.effective_chat.id, action_type, stop_event)
        )

        try:
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
                await update.message.reply_text("⚠️ Unsupported format.")
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

            # Memory (scoped to this Telegram user)
            history = await self.short_memory.get_history_messages(telegram_user_id)
            messages = [{"role": "system", "content": self.system_prompt}] + \
                    history + \
                    [{"role": "user", "content": user_content}]

            # Inference
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            response = await self.ask(messages, use_tools=True)

            if await self._send_media_if_present(update, context, response):
                return

            self.short_memory.append(telegram_user_id, user_text, response)
            self.graph_memory.append(telegram_user_id, user_text, response)  # shadow mode

            await self._send_long_message(update, response)
        except Exception as e:
            logging.exception(f"❌ Internal inference error: {str(e)}")
            await update.message.reply_text("❌ Error with vision processing.")
        finally:
            stop_event.set()
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass

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
                raise ValueError("Whisper client is not initialized.")

            chunks = self._split_audio_if_needed(local_path, max_duration_sec=60)
            
            async def _transcribe_chunk(index: int, chunk_path: str) -> tuple[int, str, str]:
                def _read_audio(path):
                    with open(path, "rb") as f:
                        return f.read()

                audio_bytes = await asyncio.get_running_loop().run_in_executor(None, _read_audio, chunk_path)
                audio_buffer = io.BytesIO(audio_bytes)
                audio_buffer.name = os.path.basename(chunk_path)
                
                logging.info(f"⚡ Starting parallel transcription for chunk {index}: {audio_buffer.name}")
                transcription = await self.whisper_client.audio.transcriptions.create(
                    model=self.env.whisper_model,
                    file=audio_buffer
                )
                return index, transcription.text.strip(), chunk_path

            tasks = [_transcribe_chunk(idx, path) for idx, path in enumerate(chunks)]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])
            
            transcriptions = []
            for _, text, chunk_path in results:
                if text:
                    transcriptions.append(text)
                if chunk_path != local_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception as e:
                        logging.warning(f"⚠️ Cannot remove temporary chunk {chunk_path}: {e}")

            transcribed_text = " ".join(transcriptions).strip()
            logging.info(f"🎙️ [WHISPER TRANSCRIPTION]: '{transcribed_text}'")

            if not transcribed_text:
                await update.message.reply_text("🔇 Cannot understand the audio.")
                return None

            return f"[Audio: {local_path}] - {transcribed_text}"

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
            logging.error(f"📸 Error processing Vision from Telegram: {e}")
            await update.message.reply_text("❌ Cannot process or store the visual file you sent.")
            return None, user_text

    async def _send_media_if_present(self, update: Update, context: ContextTypes.DEFAULT_TYPE, response: str) -> bool:
        """Sends a locally generated image or audio attachment if the response includes a media path."""
        image_match = re.search(r"(?:storage/)?images/[\w\-_]+\.png", response)
        if image_match:
            detected_path = image_match.group(0)
            if os.path.exists(detected_path):
                logging.info(f"📸 Detected image path in response: {detected_path}. Sending photo to Telegram.")
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
                with open(detected_path, "rb") as photo:
                    await update.message.reply_photo(photo=photo)
                return True

        audio_match = re.search(r"(?:storage/)?audio/([\w\-_]+\.mp3)", response)
        if audio_match:
            detected_audio_path = audio_match.group(0)
            if os.path.exists(detected_audio_path):
                logging.info(f"📁 Sending voice response: {detected_audio_path}")
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="record_voice")
                with open(detected_audio_path, "rb") as audio_file:
                    await update.message.reply_voice(voice=audio_file)
                return True

        return False

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4096):
        """Sends long messages by splitting them automatically while safely keeping Markdown code blocks intact."""
        if not text:
            return

        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        chunks = []
        current_chunk = ""
        in_code_block = False
        current_language = ""

        lines = text.split('\n')

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                if in_code_block:
                    current_language = line.strip()[3:].strip()
                else:
                    current_language = ""

            if len(current_chunk) + len(line) + 50 > max_length:
                if current_chunk:
                    if in_code_block:
                        current_chunk += "\n```"
                    chunks.append(current_chunk.strip())
                
                if in_code_block:
                    current_chunk = f"``` {current_language}\n{line}"
                else:
                    current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk:
            if in_code_block and not current_chunk.strip().endswith("```"):
                current_chunk += "\n```"
            chunks.append(current_chunk.strip())

        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(f"({i+1}/{len(chunks)})\n\n{chunk}")
                
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logging.error(f"Error sending chunk {i}: {e}")
                await update.message.reply_text(f"⚠️ The response is too long. Here's a part: {chunk[:2500]}...")

    async def _keep_alive_chat_action(self, bot, chat_id: int, action: str, stop_event: asyncio.Event):
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception as e:
                logging.debug(f"⚠️ Keep-alive action failed: {e}")
            await asyncio.sleep(4.0)

    def _split_audio_if_needed(self, local_path: str, max_duration_sec: int = 60) -> list[str]:
        try:
            audio = AudioSegment.from_file(local_path)
            duration_sec = len(audio) / 1000.0
            
            if duration_sec <= max_duration_sec:
                return [local_path]
            
            logging.info(f"🔪 The audio of {duration_sec:.1f}s exceeds the limit of {max_duration_sec}s. Splitting...")
            chunks = []
            ext = os.path.splitext(local_path)[1][1:]
            
            for i in range(0, len(audio), max_duration_sec * 1000):
                chunk = audio[i:i + max_duration_sec * 1000]
                chunk_path = f"{local_path}_part{i}.{ext}"
                chunk.export(chunk_path, format=ext)
                chunks.append(chunk_path)
                
            return chunks
        except Exception as e:
            logging.warning(f"⚠️ Cannot split the audio ({e}).")
            return [local_path]

    async def _warm_up_graphiti(self, application):
        await GraphitiClientSingleton.get(self.env)

    def run(self):
        class AuthorizedUserFilter(filters.MessageFilter):
            def __init__(self, security_manager):
                super().__init__()
                self.security_manager = security_manager
            def filter(self, message):
                return message.from_user is not None and self.security_manager.is_allowed(message.from_user.id)

        class TokenMatchFilter(filters.MessageFilter):
            def __init__(self, token):
                super().__init__()
                self.token = token
            def filter(self, message):
                return message.text is not None and message.text.strip() == self.token

        is_authorized = AuthorizedUserFilter(self.security_manager)
        is_token = TokenMatchFilter(self.env.telegram_registration_token)

        app = (
            ApplicationBuilder()
            .token(self.env.telegram_token)
            .request(HTTPXRequest(
                connection_pool_size=10,
                read_timeout=120.0,
                write_timeout=120.0,
                connect_timeout=15.0,
                pool_timeout=10.0
            ))
            .post_init(self._warm_up_graphiti)
            .build()
        )
        app.add_handler(CommandHandler("start", self.start_cmd))

        app.add_handler(MessageHandler(
            ~is_authorized & is_token & filters.TEXT & ~filters.COMMAND, 
            self.handle_registration
        ))

        app.add_handler(MessageHandler(
            is_authorized & (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO) & ~filters.COMMAND, 
            self.process
        ))

        app.add_handler(MessageHandler(
            ~is_authorized & ~filters.COMMAND, 
            self.handle_unauthorized
        ))

        app.add_error_handler(self.error_handler)

        app.run_polling()


def main():
    SethLoggerInit()
    env = SethEnvironment()
    env.validate()

    # force init
    Mem0MemorySingleton.get(env) 

    memory_tool = SethMemoryTool(env=env)
    search_tool = SethSearchTool()
    inspector_tool = SethSelfInspectorTool()
    image_tool = SethImageGenerationTool(env=env)
    speech_tool = SethSpeechGenerationTool(env=env)
    graph_query_tool = SethGraphQueryTool(env=env)

    tools_manager = SethToolsManager()
    for tool_instance in (search_tool, memory_tool, inspector_tool, image_tool, speech_tool, graph_query_tool):
        tools_manager.register_instance(tool_instance)

    vllm_client = AsyncOpenAI(base_url=env.vllm_url, api_key=env.api_key)
    whisper_client = AsyncOpenAI(base_url=env.whisper_url, api_key=env.api_key)
    current_state = replace(SethStatePresets.DEFAULT)
    current_state.load(env)
    regulator = SethDynamicRegulator(state=current_state, env=env)

    security_manager = SethSecurityBoss(env=env)

    bot_ui = SethTelegramBot(client=vllm_client, env=env, tools_manager=tools_manager, regulator=regulator, memory_tool=memory_tool, whisper_client=whisper_client, security_manager=security_manager)
    bot_ui.run()


if __name__ == '__main__':
    main()