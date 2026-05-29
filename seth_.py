"""
====================================================================================================
PROJECT: SETH-IN-A-BOX (MVP-2)
DATE: May 27, 2026
ARCHITECT: Luis Capra (@luis.capra)

DESCRIPTION:
    This module bridges the gap between static LLM execution and autonomous cognitive continuity.
    By fusing a highly optimized local vLLM pipeline (Gemma-4-26B FP4) with an asynchronous 
    distributed web-crawling engine (Crawl4AI) and a vectorized long-term memory fabric (Qdrant),
    SETH-in-a-Box MVP-2 transcends the limitations of transient context windows.

    This architecture morphs the traditional chatbot paradigm into a persistent, self-evolving 
    dialectic companion capable of real-time web ingestion, contextual semantic synthesis, and 
    cross-session recollection.

    "The noise of the void, structured into code."

fix: MVP-2 stabilization TODOs before merge

CRITICAL (breaks runtime):
- [ ] SethTelegramBot references self.embedding_engine which doesn't exist
      → inject it via constructor or pull from SethMemoryTool.memory.embedding_model

HIGH (breaks multi-user):
- [ ] user_id hardcoded as 123 in top-level tool functions (web_search, retrieve_long_term_memory, save_long_term_memory)
      → thread user_id through tool call context or use a per-request closure

MEDIUM (silent failures):
- [ ] save_long_term_memory is a registered tool but the LLM decides when to call it
      → consider forcing a save after every meaningful exchange or add an explicit post-hook

LOW (nice to have):
- [ ] SethMemoryTool is re-instantiated on every tool call (cold Qdrant connection each time)
      → lift instance to bot constructor and reuse
- [ ] SethEnvironment.validate() is called inside run() but client/tools are built before it
      → move validate() call to main() before constructing anything
      
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

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from ddgs import DDGS
from dotenv import load_dotenv
from mem0 import Memory
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from openai import OpenAI
from collections import deque

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
        """Fetch max_results URLs from DuckDuckGo and crawl them concurrently using asyncio.gather."""
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
        """Crawl a single URL under the semaphore. Returns (url, res, crawl) or None."""
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
    def __init__(self, user_id: int, collection_name: str, env: SethEnvironment):
        self.env = env
        self.user_id = user_id
        self.collection_name = collection_name

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
        """Retrieve and format long-term memory entries asynchronously without blocking the loop."""
        def _sync_search():
            return self.memory.search(
                query,
                filters={"user_id": str(self.user_id)},
                limit=10,
            )

        try:
            raw_retrieval = await asyncio.to_thread(_sync_search)
            
            results = raw_retrieval if isinstance(raw_retrieval, list) else raw_retrieval.get("results", [])
            records = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            facts = [f"- {r['memory'].strip()}" for r in records if r.get("memory")]
            long_term_context = "\n".join(facts) if facts else "No historical records found for this interlocutor."

        except Exception as e:
            logging.warning(f"Error retrieving memories for user {self.user_id}: {e}")
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
                user_id=str(self.user_id),
                agent_id="SETH",
                metadata={"memory_bucket": "constraints", "expires_on": expiration},
            )

        try:
            res = await asyncio.to_thread(_sync_add)
            return {"status": "ok", "result": res}
        except Exception as e:
            logging.exception(f"Failed to save memory for user {self.user_id}: {e}")
            return {"status": "error", "error": str(e)}


class ToolsManager:
    """Registry for function-calling tools: registration, validation and
    serialization into the vLLM/OpenAI `tools` JSON shape.
    """
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
        if not isinstance(tool, dict):
            raise TypeError("tool must be a dict")
        if "name" not in tool or "parameters" not in tool:
            raise ValueError("tool definition must include 'name' and 'parameters'")

    def as_vllm_format(self) -> list[dict]:
        """Return tools wrapped as vLLM/OpenAI expects: list of {type:function, function:...}"""
        out = []
        for t in self._tools:
            self._validate_tool(t)
            out.append({"type": "function", "function": t})
        return out

    def get_function(self, name: str):
        return self._funcs.get(name)


MEMORY = SethMemoryTool(user_id=123, collection_name="TEST", env=SethEnvironment())
SEARCH = SethSearchTool()

async def retrieve_long_term_memory(query):
    logging.info(f"🌐 Retrieving long-term memory for query: {query}")
    return await MEMORY.retrieve_long_term_memory(query=query)


async def save_long_term_memory(user_input: str, response:str):
    logging.info(f"🌐 Saving long-term memory for user_input: {user_input}, response: {response}")
    return await MEMORY.save_long_term_memory(user_input=user_input, response=response)

async def web_search(query):
    logging.info(f"🌐 Crawling the web for: {query}")
    return await SEARCH.search(query)


class ToolRegistry:
    """Encapsulates tool registration so registrations can be organized elsewhere."""
    def __init__(self, tools_manager: ToolsManager):
        self.tools_manager = tools_manager

    def register_default_tools(self):
        self.tools_manager.register(
            "web_search",
            web_search,
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
        self.tools_manager.register(
            "retrieve_long_term_memory",
            retrieve_long_term_memory,
            "Retrieves long-term memory for a user. Useful for recalling past interactions, preferences, and important facts about the user. useful for resolve ambiguous questions about the user or their preferences, and for maintaining continuity across interactions.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        self.tools_manager.register(
            "save_long_term_memory",
            save_long_term_memory,
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


class SethChatBot:
    """Wraps the OpenAI client and orchestrates chat calls and optional tools.
    
    Completes the execution loop by feeding tool results back to the LLM.
    """
    def __init__(self, client: OpenAI, tools_manager: ToolsManager | None = None):
        self.client = client
        self.tools_manager = tools_manager
        self.env = SethEnvironment()

    async def send(self, user_text: str, use_tools: bool = True) -> str:
        messages = [{"role": "user", "content": user_text}]
        return await self.ask(messages, use_tools=use_tools)

    async def ask(self, messages: str, use_tools: bool = True) -> str:
        tools = self.tools_manager.as_vllm_format() if (use_tools and self.tools_manager) else None

        def _sync_call(messages):
            logging.info("🧠 Sending message to LLM with tool options = [{tools}]".format(tools="Yes" if tools else "No"))

            if tools:
                return self.client.chat.completions.create(
                    model=self.env.llm_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

            return self.client.chat.completions.create(
                model=self.env.llm_model,
                messages=messages
            )

        response = await asyncio.to_thread(_sync_call, messages)
        choice = response.choices[0]
        message = choice.message
        
        messages.append(message)

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

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": str(result)
                })

            logging.info("🧠 Feeding tool results back to the LLM for final synthesis...")
            final_response = await asyncio.to_thread(_sync_call, messages)
            return final_response.choices[0].message.content

        return message.content
    

class SethShortMemory:
    """Manages short-term conversational context window using an atomic sliding queue."""
    def __init__(self, env: SethEnvironment, max_history: int = 40):
        self.env = env
        # each exchange has 2 entries (user + assistant)
        self._history = deque(maxlen=max_history * 2)

    def system_prompt(self) -> str:
        """Loads SETH's core ontological identity configuration."""
        if os.path.exists(self.env.system_prompt_path):
            try:
                with open(self.env.system_prompt_path, "r", encoding="utf-8") as f:
                    return f"<SYSTEM>\n{f.read()}\n</SYSTEM>"
            except Exception as e:
                logging.error(f"❌ Error reading system prompt: {e}")
        
        return "<SYSTEM>\nPLEASE TELL THE USER TO PROVIDE THEIR SYSTEM PROMPT.\n</SYSTEM>"
    
    def get_history_messages(self) -> list[dict]:
        """Returns a safe, serializable list of past conversation steps for vLLM."""
        return list(self._history)

    def append(self, text: str, response: str):
        """Atomically pushes the latest exchange into the sliding window."""
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": response})
        
    def clear(self):
        """Flushes the short term memory context."""
        self._history.clear()


class SethTelegramBot(SethChatBot):
    """Bridges SethChatBot logic directly to Telegram events via inheritance."""
    def __init__(self, client: OpenAI, tools_manager: ToolsManager | None = None):
        super().__init__(client, tools_manager)
        self.short_memory = SethShortMemory(self.env)
        self.system_prompt = self.short_memory.system_prompt()
        
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- SETH ONLINE (NOT INTENDED FOR MULTIPLE USERS OR PRODUCTION USE)---")

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Intercepts all incoming text messages from Telegram and routes through SETH."""
        user_text = update.message.text

        messages = [{"role": "system", "content": self.system_prompt}] + self.short_memory.get_history_messages() + [{"role": "user", "content": user_text}]

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            response = await self.ask(messages, use_tools=True)
            self.short_memory.append(user_text, response)
            await self._send_message(update, response)
        except Exception as e:
            logging.exception("Error during seth real-time processing loop.")
            await update.message.reply_text(f"❌ Error interno en la inferencia de SETH: {str(e)}")

    async def _send_message(self, update: Update, text: str, max_length: int = 4000):
        """Sends long messages by splitting them automatically."""
        if not text:
            return

        # If short -> send normally
        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        # Split into chunks
        chunks = []
        current_chunk = ""
        
        for paragraph in text.split('\n'):
            if len(current_chunk) + len(paragraph) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n" + paragraph if current_chunk else paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Send all chunks
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(f"({i+1}/{len(chunks)})\n\n{chunk}")
                
                # Small pause to avoid spamming
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logging.error(f"Error sending chunk {i}: {e}")
                # Fallback attempt
                await update.message.reply_text("⚠️ The response is too long. Here's a part:")
                await update.message.reply_text(chunk[:3500])

    def run(self):
        """Starts the asynchronous Telegram polling application."""
        self.env.validate()
        app = ApplicationBuilder().token(self.env.telegram_token).build()
        app.add_handler(CommandHandler("start", self.start_cmd))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.process))
        logging.info("🚀 SETH UI Pipeline running. Waiting for telegram updates...")
        app.run_polling()


def main():
    env = SethEnvironment()
    client = OpenAI(base_url=env.vllm_url, api_key="dummy")
    tools_manager = ToolsManager()
    ToolRegistry(tools_manager).register_default_tools()
    bot_ui = SethTelegramBot(client, tools_manager)
    bot_ui.run()


if __name__ == '__main__':
    main()
