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
- [ ] SethDynamicRegulator is instantiated but never wired to SethChatBot.ask()
      → pass regulator.update(query) result as extra kwargs to the LLM call

HIGH (breaks multi-user):
- [ ] user_id hardcoded as 123 in top-level tool functions (web_search, retrieve_long_term_memory, save_long_term_memory)
      → thread user_id through tool call context or use a per-request closure

MEDIUM (silent failures):
- [ ] SethDynamicRegulator.update() result (temperature, top_p, etc.) is computed but discarded
      → apply returned config dict to the completions.create() call
- [ ] _send_long_message() is defined in SethTelegramBot but never called in process()
      → replace reply_text(seth_response) with _send_long_message()
- [ ] save_long_term_memory is a registered tool but the LLM decides when to call it
      → consider forcing a save after every meaningful exchange or add an explicit post-hook

LOW (nice to have):
- [ ] SethMemoryTool is re-instantiated on every tool call (cold Qdrant connection each time)
      → lift instance to bot constructor and reuse
- [ ] No conversation history passed to SethChatBot.ask() across turns
      → add short-term history deque similar to MVP-1 SethMemory
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
    soul_path: str = "seth.md"
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))

    def validate(self):
        if not self.telegram_token:
            raise ValueError("❌ TELEGRAM_TOKEN is missing in the environment.")


async def web_search(query):
    logging.info(f"🌐 Crawling the web for: {query}")
    return await SethSearchTool().search(query)


async def retrieve_long_term_memory(query):
    user_id = 123
    logging.info(f"🌐 Retrieving long-term memory for user: {user_id}, query: {query}")
    memory = SethMemoryTool(user_id=user_id, collection_name="TEST", env=SethEnvironment())
    return memory.retrieve_long_term_memory(query=query)


async def save_long_term_memory(user_input: str, response:str):
    user_id = 123
    logging.info(f"🌐 Saving long-term memory for user: {user_id}, user_input: {user_input}, response: {response}")
    memory = SethMemoryTool(user_id=user_id, collection_name="TEST", env=SethEnvironment())
    return memory.save_long_term_memory(user_input=user_input, response=response)


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
                "config": {"model": env.llm_model, "openai_base_url": env.vllm_url, "api_key": "NONE"}
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
            - Greetings, conversational filler, and casual chatter ("cool", "thanks", "ok", "che").
            - Transient debugging steps, syntax errors, or temporary trial-and-error logs, UNLESS explicitly stated as a definitive fix or final architecture.
            - Redundant or duplicate facts already established in the conversation history.
            
            Return JSON with key "facts" as a list of strings (use [] if nothing to store).
            """
        }
        self.memory = Memory.from_config(mem0_config)

    def retrieve_long_term_memory(self, query: str) -> str:
        """Retrieve and format long-term memory entries for a user."""
        try:
            raw_retrieval = self.memory.search(
                query,
                filters={"user_id": str(self.user_id)},
                limit=10,
            )
            
            records = sorted(raw_retrieval.get("results", []), key=lambda x: x.get("score", 0), reverse=True)
            
            facts = [f"- {r['memory'].strip()}" for r in records if r.get("memory")]
            long_term_context = "\n".join(facts) if facts else "No historical records found for this interlocutor."

        except Exception as e:
            logging.warning(f"Error retrieving memories for user {self.user_id}: {e}")
            long_term_context = "Could not retrieve long-term memory."

        return f"<MEMORY>\n{long_term_context}\n</MEMORY>\n\n"

    def save_long_term_memory(self, user_input: str, response: str) -> Dict[str, Any]:
        """Persist a memory item to the vector store.

        Returns a dict with operation status and stored record metadata.
        """
        try:
            expiration = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
            res =self.memory.add(
                    [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response},
                    ],
                    user_id=str(self.user_id),
                    agent_id="SETH",
                    metadata={"memory_bucket": "constraints", "expires_on": expiration},
                )

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
    """Orchestrates semantic state shifts based on real-time query intent."""
    def __init__(self, state: SethState, embedding_engine, alpha=0.2):
        self.embedding_engine = embedding_engine
        self.alpha = alpha
        self.current_state = state 
        
        # Reference vectors for SETH's behavioral modes
        self.targets = {
            "rigorous": {
                "vector": self.embedding_engine.encode("Technical architecture, code precision, logic, systems design"),
                "state": SethPresets.rigorous()
            },
            "chaotic": {
                "vector": self.embedding_engine.encode("Glitch aesthetics, humor, creative chaos, fertile glitch"),
                "state": SethPresets.chaotic()
            },
            "verbose": {
                "vector": self.embedding_engine.encode("Deep philosophy, ontological analysis, long essay, legacy"),
                "state": SethPresets.verbose()
            }
        }

    def update(self, query: str) -> Dict[str, Any]:
        """Calculates the best-fit behavioral mode and updates the current state."""
        q_vec = self.embedding_engine.encode(query)
        best_name, _ = max(
            ((n, np.dot(q_vec, d["vector"]) / (np.linalg.norm(q_vec) * np.linalg.norm(d["vector"]))) 
             for n, d in self.targets.items()), 
            key=lambda x: x[1]
        )
        self.current_state.interpolate(self.targets[best_name]["state"], self.alpha)
        logging.info(f"🌀 STATE ADJUSTMENT - New Temp: {self.current_state.temperature:.3f} | Top_p: {self.current_state.top_p:.3f}")

        return self.current_state.to_dict()


class SethTelegramBot(SethChatBot):
    """Bridges SethChatBot logic directly to Telegram events via inheritance."""
    def __init__(self, client: OpenAI, tools_manager: ToolsManager | None = None):
        super().__init__(client, tools_manager)
        self.regulator: SethDynamicRegulator = SethDynamicRegulator(SethPresets.DEFAULT, self.embedding_engine)
        
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- SETH ONLINE (NOT INTENDED FOR MULTIPLE USERS OR PRODUCTION USE)---")

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Intercepts all incoming text messages from Telegram and routes through SETH."""
        user_text = update.message.text

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            seth_response = await self.send(user_text, use_tools=True)
            await update.message.reply_text(seth_response)
        except Exception as e:
            logging.exception("Error during seth real-time processing loop.")
            await update.message.reply_text(f"❌ Error interno en la inferencia de SETH: {str(e)}")

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000):
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
