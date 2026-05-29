"""
====================================================================================================
PROJECT: SETH-IN-A-BOX (MVP)
DATE: May 28, 2026

DESCRIPTION:
    This module bridges the gap between static LLM execution and autonomous cognitive continuity.
    By fusing a highly optimized local vLLM pipeline (Gemma-4-26B FP4) with an asynchronous 
    distributed web-crawling engine (Crawl4AI) and a vectorized long-term memory fabric (Qdrant),
    SETH-in-a-Box MVP transcends the limitations of transient context windows.

    "The noise of the void, structured into code."

STATUS: 
    - Asynchronous & Non-blocking execution loop via asyncio.to_thread wrappers.
    - Persistent Qdrant connection management (Lifting instance to Bot scope).
    - Isolated history array mutations to prevent race conditions or schema contamination.
    - Clean Environment validation before bootstrapping components.
====================================================================================================
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
import os
import re
import time
from typing import Any, Dict
from collections import deque

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from ddgs import DDGS
import numpy as np
from dotenv import load_dotenv
from mem0 import Memory
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from openai import OpenAI
from sentence_transformers import SentenceTransformer

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
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_dims: int = int(os.getenv("EMBEDDING_MODEL_DIMS", 1024))
    system_prompt_path: str = "seth.md"
    log_path: str = "conversation_history.jsonl"
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
                    page_timeout=5000,
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


class SethMemoryTool:
    """Multi-layered memory manager integrated with Qdrant Vector Store."""
    def __init__(self, collection_name: str, env: SethEnvironment):
        self.env = env
        self.collection_name = collection_name
        self.static_user_id = "seth_core_user"

        mem0_config = {
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
            "custom_instructions": """
            Extract from the system architecture, programming, and research conversations:
            - Core conceptual definitions, original theories, and philosophical theses.
            - High-level software engineering patterns, system design decisions, hardware configurations, and infrastructure specs.
            - Explicit user preferences, long-term project goals, and definitive facts about the user's workflow or environment.
            
            Exclude:
            - Transient debugging steps, syntax errors, or temporary trial-and-error logs, UNLESS explicitly stated as a definitive fix or final architecture.
            - Redundant or duplicate facts already established in the conversation history.
            
            Return JSON with key "facts" as a list of strings (use [] if nothing to store).
            """
        }
        self.memory = Memory.from_config(mem0_config)

    async def retrieve_long_term_memory(self, query: str) -> str:
        """Retrieve and format long-term memory entries asynchronously without blocking."""
        def _sync_search():
            return self.memory.search(
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

    async def save_long_term_memory(self, user_input: str, response: str) -> Dict[str, Any]:
        """Persist a memory item to the vector store asynchronously."""
        expiration = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        def _sync_add():
            return self.memory.add(
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SethPresets:
    """Standardized behavioral states for the SETH ecosystem."""
    
    # Static presets for quick access
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
    def __init__(self, state: Any, model_path: str = "BAAI/bge-large-en-v1.5", alpha: float = 0.2):
        self.alpha = alpha
        self.current_state = state 
        
        logging.info(f"💾 Loading Embedding Engine from: {model_path}")
        self.engine = SentenceTransformer(model_path)
        
        self.targets = {
            "rigorous": {
                "vector": self.engine.encode("Technical architecture, code precision, logic, systems design", normalize_embeddings=True),
                "state": SethPresets.rigorous()
            },
            "chaotic": {
                "vector": self.engine.encode("Glitch aesthetics, humor, creative chaos, fertile glitch", normalize_embeddings=True),
                "state": SethPresets.chaotic()
            },
            "verbose": {
                "vector": self.engine.encode("Deep philosophy, ontological analysis, long essay, legacy", normalize_embeddings=True),
                "state": SethPresets.verbose()
            }
        }

    def regulate(self, query: str) -> Dict[str, Any]:
        """Calculates the best-fit behavioral mode and updates the current state."""
        q_vec = self.engine.encode(query, normalize_embeddings=True)
        
        best_name, _ = max(
            ((n, np.dot(q_vec, d["vector"])) for n, d in self.targets.items()), 
            key=lambda x: x[1]
        )
        self.current_state.interpolate(self.targets[best_name]["state"], self.alpha)
        logging.info(f"🌀 STATE ADJUSTMENT [{best_name.upper()}] - New Temp: {self.current_state.temperature:.3f} | Top_p: {self.current_state.top_p:.3f}")

        return self.current_state.to_dict()
    
class SethChatBot:
    """Wraps the OpenAI client and orchestrates async chat calls and execution loop."""
    def __init__(self, client: OpenAI, env: SethEnvironment, tools_manager: ToolsManager | None = None, regulator: SethDynamicRegulator | None = None):
        self.client = client
        self.env = env
        self.tools_manager = tools_manager
        self.memory_tool = SethMemoryTool(collection_name="SETH_CORE_SPACE", env=env)
        self.regulator = regulator

    async def ask(self, messages: list[dict], use_tools: bool = True) -> str:
        tools = self.tools_manager.as_vllm_format() if (use_tools and self.tools_manager) else None
        local_messages = list(messages)

        if local_messages:
            last_user_query = local_messages[-1].get("content", "")
            
            if last_user_query:
                related_context = await self.memory_tool.retrieve_long_term_memory(last_user_query)
                local_messages[-1] = {
                    "role": local_messages[-1]["role"],
                    "content": f"{related_context}\n\n{last_user_query}"
                }

        def _sync_call(payload):
            logging.info(f"🧠 Sending message to vLLM.")
            config = self.regulator.regulate(last_user_query) if self.regulator else {}
            if tools:
                return self.client.chat.completions.create(
                    model=self.env.llm_model, messages=payload, tools=tools, tool_choice="auto", **config
                )
            return self.client.chat.completions.create(model=self.env.llm_model, messages=payload, **config)

        response = await asyncio.to_thread(_sync_call, local_messages)
        choice = response.choices[0]
        message = choice.message
        
        local_messages.append(message)

        if getattr(message, 'tool_calls', None):
            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}

                logging.info(f"🛠️ Executing tool requested by LLM: {name} with args: {args}")
                fn = self.tools_manager.get_function(name) if (self.tools_manager and name) else None
                
                try:
                    if fn:
                        maybe_coro = fn(**args)
                        result = await maybe_coro if asyncio.iscoroutine(maybe_coro) else maybe_coro
                    else:
                        result = f"Error: Tool '{name}' not found."
                except Exception as exc:
                    result = f"Error executing tool: {str(exc)}"

                local_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": str(result)
                })

            logging.info("🧠 Feeding tool results back to the LLM for final synthesis...")
            final_response = await asyncio.to_thread(_sync_call, local_messages)

            return self.clean_channel_tags(final_response.choices[0].message.content)

        return message.content

    def clean_channel_tags(self, text: str) -> str:
        """ Temporary utility to clean up channel tags from the vLLM output after using tools. """
        if not text:
            return ""
        
        pattern = r'(<\s*\|?\s*channel\s*\|?\s*>).*?(<\s*\|?\s*channel\s*\|?\s*>)'
        
        return re.sub(pattern, '💾 ', text, flags=re.DOTALL | re.IGNORECASE)


class SethShortMemory:
    """Manages short-term conversational context window using an atomic sliding queue and persists logs."""
    def __init__(self, env: SethEnvironment, max_history: int = 40):
        self.env = env
        self._history = deque(maxlen=max_history * 2)

    def system_prompt(self) -> str:
        if os.path.exists(self.env.system_prompt_path):
            try:
                with open(self.env.system_prompt_path, "r", encoding="utf-8") as f:
                    return f"<SYSTEM>\n{f.read()}\n</SYSTEM>"
            except Exception as e:
                logging.error(f"❌ Error reading system prompt: {e}")
        return "<SYSTEM>\nREQUEST THE USER TO PROVIDE A VALID SYSTEM PROMPT!.\n</SYSTEM>"
    
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
            try:
                with open(self.env.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.error(f"❌ Error writing transaction to JSONL: {e}")

        await asyncio.to_thread(_sync_write)


class SethTelegramBot(SethChatBot):
    """Bridges SethChatBot logic to Telegram events using clean state management."""
    def __init__(self, client: OpenAI, env: SethEnvironment, tools_manager: ToolsManager | None = None):
        super().__init__(client, env, tools_manager)
        self.short_memory = SethShortMemory(self.env)
        self.system_prompt = self.short_memory.system_prompt()
        
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- SETH ONLINE ---")

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        messages = [{"role": "system", "content": self.system_prompt}] + \
                   self.short_memory.get_history_messages() + \
                   [{"role": "user", "content": user_text}]

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            response = await self.ask(messages, use_tools=True)
            self.short_memory.append(user_text, response)
            await self._send_message(update, response)
        except Exception as e:
            logging.exception("Error during seth real-time processing loop.")
            await update.message.reply_text(f"❌ Error interno de inferencia: {str(e)}")

    async def _send_message(self, update: Update, text: str, max_length: int = 4000):
        if not text: return
        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        chunks = []
        current_chunk = ""
        for paragraph in text.split('\n'):
            if len(current_chunk) + len(paragraph) + 1 > max_length:
                if current_chunk: chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n" + paragraph if current_chunk else paragraph
        if current_chunk: chunks.append(current_chunk.strip())

        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(f"({i+1}/{len(chunks)})\n\n{chunk}")
                await asyncio.sleep(0.3)
            except Exception as e:
                logging.error(f"Error sending chunk {i}: {e}")

    def run(self):
        app = ApplicationBuilder().token(self.env.telegram_token).build()
        app.add_handler(CommandHandler("start", self.start_cmd))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.process))
        logging.info("🚀 SETH Pipeline Up. Polling updates...")
        app.run_polling()


def main():
    env = SethEnvironment()
    env.validate()

    memory_tool = SethMemoryTool(collection_name="SETH_CORE_SPACE", env=env)
    search_tool = SethSearchTool()

    tools_manager = ToolsManager()
    
    tools_manager.register(
        "web_search",
        search_tool.search,
        "Executes a live web search to fetch real-time information, current events, up-to-date technical documentation, or factual updates. Use this tool whenever the query requires data beyond your knowledge cutoff, recent market conditions, or verification of breaking news.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The precise search query. Use concise, targeted keywords. Do NOT include conversational filler like 'search for', 'find info about', or punctuation."
                }
            },
            "required": ["query"],
        },
    )
    tools_manager.register(
        "retrieve_long_term_memory",
        memory_tool.retrieve_long_term_memory,
        "Retrieves long-term memory for a user. Useful for recalling past interactions, preferences, and important facts about the user. useful for resolve ambiguous questions about the user or their preferences, and for maintaining continuity across interactions.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    tools_manager.register(
        "save_long_term_memory",
        memory_tool.save_long_term_memory,
        "Persists meaningful facts, constraints, project decisions, or user preferences from the current exchange into long-term memory. Use this tool ONLY when the interaction contains important details that should be remembered across future sessions. Do NOT use it for generic greetings or transient chatter.",
        {
            "type": "object",
            "properties": {
                "user_input": {
                    "type": "string", 
                    "description": "The exact user message containing the core fact, preference, or constraint to persist."
                },
                "response": {
                    "type": "string", 
                    "description": "The assistant response that validates, confirms, or completes the memory context."
                }
            },
            "required": ["user_input", "response"],
        },
    )

    client = OpenAI(base_url=env.vllm_url, api_key="dummy")
    regulator = SethDynamicRegulator(state=SethPresets.DEFAULT)

    bot_ui = SethTelegramBot(client=client, env=env, tools_manager=tools_manager, regulator=regulator)
    bot_ui.run()


if __name__ == '__main__':
    main()