import os
import logging
import re
import json
import copy
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import asyncio

import openai
from mem0 import Memory
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from ddgs import DDGS
from crawl4ai import AsyncWebCrawler

# Initial System Configuration
load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

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

@dataclass(frozen=True)
class SethEnvironment:
    """Centralized and typed configuration for the SETH ecosystem."""
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
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


class SethSearchTool:
    """Web retrieval and extraction tool powered by DuckDuckGo and Crawl4AI."""
    async def search(self, query: str, max_results: int = 5) -> str:
        logging.info(f"🌐 Crawling the web for: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, timelimit="m"))

            if not results:
                return "<SEARCH_ERROR>No results.</SEARCH_ERROR>"

            context_str = "\n<WEB_SEARCH_RESULTS>\n"
            async with AsyncWebCrawler() as crawler:
                for res in results:
                    url = res.get('href')
                    if not url: continue
                    
                    crawl = await crawler.arun(url=url, bypass_cache=True, timeout=1000)
                    if crawl and crawl.success:
                        # Normalize whitespace and truncate content to avoid context bloat
                        content = re.sub(r'\s+', ' ', crawl.markdown[:2500])
                        context_str += f"Source: {url}\nContent: {content}\n\n"
            
            return context_str + "</WEB_SEARCH_RESULTS>"
        except Exception as e:
            return f"<SEARCH_ERROR>{str(e)}</SEARCH_ERROR>"

class SethDynamicRegulator:
    """Orchestrates semantic state shifts based on real-time query intent."""
    def __init__(self, state: SethState, bge_engine, alpha=0.2):
        self.engine = bge_engine
        self.alpha = alpha
        self.current_state = state 
        
        # Reference vectors for SETH's behavioral modes
        self.targets = {
            "rigorous": {
                "vector": self.engine.encode("Technical architecture, code precision, logic, systems design"),
                "state": SethPresets.rigorous()
            },
            "chaotic": {
                "vector": self.engine.encode("Glitch aesthetics, humor, creative chaos, fertile glitch"),
                "state": SethPresets.chaotic()
            },
            "verbose": {
                "vector": self.engine.encode("Deep philosophy, ontological analysis, long essay, legacy"),
                "state": SethPresets.verbose()
            }
        }

        search_trigger_text = """
            Current news, weather, prices, real-time events, latest info,
            current events, search, web search, current, recent, web,
            internet, price, weather, recent, latest, external, search
            the internet, search web, web search, today, yesterday
        """.replace("\n", "").replace("  ", " ").strip()

        self.search_trigger = self.engine.encode(search_trigger_text)

    def should_search(self, query: str) -> bool:
        """Determines if the query requires external web search via semantic similarity."""
        q_vec = self.engine.encode(query)
        sim = np.dot(q_vec, self.search_trigger) / (np.linalg.norm(q_vec) * np.linalg.norm(self.search_trigger))
        return sim > 0.48

    def update(self, query: str) -> Dict[str, Any]:
        """Calculates the best-fit behavioral mode and updates the current state."""
        q_vec = self.engine.encode(query)
        best_name, _ = max(
            ((n, np.dot(q_vec, d["vector"]) / (np.linalg.norm(q_vec) * np.linalg.norm(d["vector"]))) 
             for n, d in self.targets.items()), 
            key=lambda x: x[1]
        )
        self.current_state.interpolate(self.targets[best_name]["state"], self.alpha)
        logging.info(f"🌀 STATE ADJUSTMENT - New Temp: {self.current_state.temperature:.3f} | Top_p: {self.current_state.top_p:.3f}")

        return self.current_state.to_dict()

class SethMemory:
    """Multi-layered memory manager integrated with Qdrant Vector Store."""
    def __init__(self, env: SethEnvironment, max_history=30):
        self.env = env
        self.max_history = max_history
        self.user_short_term: Dict[int, deque] = {}
        
        # Connection configuration for the standalone Qdrant Server
        mem0_config = {
            "llm": {
                "provider": "openai", 
                "config": {"model": env.llm_model, "openai_base_url": env.vllm_url, "api_key": "EMPTY"}
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
                    "collection_name": "seth_memories",
                    "embedding_model_dims": env.embedding_dims
                }
            }
        }
        self.eternal_memory = Memory.from_config(mem0_config)
        self.bge_engine = self.eternal_memory.embedding_model.model

    def _normalize_mem0_results(self, raw_retrieval: Any) -> List[Dict]:
        """Normaliza el resultado de mem0.search() independientemente de la versión."""
        if raw_retrieval is None:
            return []
        
        # Caso 1: Ya es una lista (comportamiento v1.0)
        if isinstance(raw_retrieval, list):
            return raw_retrieval
        
        # Caso 2: Es un diccionario con "results" (comportamiento v1.1 + Platform)
        if isinstance(raw_retrieval, dict):
            # Prioridad al key "results"
            if "results" in raw_retrieval:
                results = raw_retrieval["results"]
                return results if isinstance(results, list) else []
            
            # Fallback: a veces devuelve directamente los items
            return [raw_retrieval] if any(k in raw_retrieval for k in ("memory", "id")) else []
        
        # Caso raro: objeto con atributo results (por si acaso)
        if hasattr(raw_retrieval, "results"):
            results = raw_retrieval.results
            return results if isinstance(results, list) else []
        
        return []

    def get_full_context(self, user_id: int, name: str, query: str) -> str:
        """Retrieves and compiles context from Soul Ontology and Eternal Memory."""
        # 1. Soul Ontology
        soul_ontology = ""
        if os.path.exists(self.env.soul_path):
            with open(self.env.soul_path, "r", encoding="utf-8") as f:
                soul_ontology = f"<SOUL_ONTOLOGY>\n{f.read()}\n</SOUL_ONTOLOGY>"

        # 2. Eternal Memory (mejor normalizado)
        try:
            raw_retrieval = self.eternal_memory.search(
                query, 
                filters={"user_id": str(user_id)}, 
                limit=10
            )
            
            memory_records = self._normalize_mem0_results(raw_retrieval)
            
            # RANK - we should improve this..
            if isinstance(memory_records, list) and memory_records:
                memory_records = sorted(
                    memory_records, 
                    key=lambda x: x.get('score', 0) if isinstance(x, dict) else 0,
                    reverse=True
                )
            
            facts = []
            for r in memory_records:
                if isinstance(r, dict):
                    memory_text = r.get('memory') or r.get('content') or str(r)
                else:
                    memory_text = str(r)
                
                if memory_text.strip():
                    facts.append(f"- {memory_text.strip()}")
                    
        except Exception as e:
            logging.warning(f"Error retrieving memories for user {user_id}: {e}")
            facts = []
            long_term_context = "No se pudo recuperar memoria a largo plazo."
        else:
            long_term_context = "\n".join(facts) if facts else "No historical records found for this interlocutor."

        return (
            f"{soul_ontology}\n\n"
            f"<LONG_TERM_MEMORY>\n{long_term_context}\n</LONG_TERM_MEMORY>\n\n"
            f"Active interlocutor: {name} (UID: {user_id})."
        )

    def get_short_term(self, user_id: int) -> deque:
        """Manages the sliding window for the current conversation history."""
        if user_id not in self.user_short_term:
            self.user_short_term[user_id] = deque(maxlen=self.max_history)
        return self.user_short_term[user_id]

class SethEngine:
    """Inference engine with Function Calling and Data Synthesis support."""
    def __init__(self, env: SethEnvironment):
        self.env = env
        self.client = openai.AsyncOpenAI(base_url=env.vllm_url, api_key="EMPTY")
        self.search_tool = SethSearchTool()

    def _get_tools(self) -> List[Dict]:
        """Defines the tool schema for vLLM function calling."""
        return [{
            "type": "function", 
            "function": {
                "name": "web_search", 
                "description": "Searches for real-time updated information on the internet.", 
                "parameters": {
                    "type": "object", 
                    "properties": {"query": {"type": "string"}}, 
                    "required": ["query"]
                }
            }
        }]

    async def _handle_search_flow(self, choice, messages, use_tools) -> Optional[str]:
        """Handles both native tool calls and semantic-triggered bypass searches."""
        if choice.message.tool_calls:
            args = json.loads(choice.message.tool_calls[0].function.arguments)
            return await self.search_tool.search(args['query'])
        
        # Fallback mechanism for models with limited Tool-Call support
        if use_tools and not choice.message.content:
            tmp_msg = messages + [{"role": "user", "content": "Return ONLY a search query in <query>...</query>"}]
            resp = await self.client.chat.completions.create(model=self.env.llm_model, messages=tmp_msg, temperature=0)
            match = re.search(r'<query>(.*?)</query>', resp.choices[0].message.content)
            if match: return await self.search_tool.search(match.group(1))
        
        return None

    async def ask(self, system_prompt: str, history: deque, user_input: str, regulator: SethDynamicRegulator) -> str:
        """Main inference loop with state regulation and tool execution."""
        config = regulator.update(user_input)
        use_tools = regulator.should_search(user_input)
        
        messages = [{"role": "system", "content": system_prompt}] + list(history) + [{"role": "user", "content": user_input}]

        try:
            resp = await self.client.chat.completions.create(
                model=self.env.llm_model,
                messages=messages,
                tools=self._get_tools() if use_tools else None,
                **config
            )

            search_data = await self._handle_search_flow(resp.choices[0], messages, use_tools)
            if search_data:
                return await self._synthesize(user_input, search_data, history)

            return resp.choices[0].message.content or "Error: Empty inference generated."

        except Exception as e:
            logging.error(f"Engine Exception: {e}")
            return f"Inference Circuit Break: {str(e)}"

    async def _synthesize(self, user_input: str, data: str, history: deque) -> str:
        """Synthesizes the final response by injecting retrieved web data into context."""
        messages = [
            {"role": "system", "content": f"You are SETH. Synthesize this data into a coherent response:\n{data}"},
            *list(history)[-3:], 
            {"role": "user", "content": user_input}
        ]
        resp = await self.client.chat.completions.create(
            model=self.env.llm_model,
            messages=messages,
            temperature=0.4
        )

        return resp.choices[0].message.content

class SethTelegramBot:
    """Telegram interface and memory orchestration layer."""
    def __init__(self, env: SethEnvironment, base_state: SethState, engine: SethEngine, memory: SethMemory):
        self.app = ApplicationBuilder().token(env.telegram_token).build()
        self.engine = engine
        self.memory = memory
        self.base_state = base_state
        self.regulators: Dict[int, SethDynamicRegulator] = {}
        
        # Bot Handlers
        self.app.add_handler(CommandHandler("start", self.start_cmd))
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.process))

    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("--- SETH ONLINE ---")

    async def process(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        u_id = update.effective_user.id
        u_name = update.effective_user.first_name
        text = update.message.text.strip()
        
        # Audit Log
        logging.info(f"--- INCOMING REQUEST ---")
        logging.info(f"User: {u_name} (ID: {u_id})")
        logging.info(f"Regulator initialized: {u_id in self.regulators}")
        logging.info(f"Short-term memory depth: {len(self.memory.get_short_term(u_id))}")

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        # Initialize per-user Dynamic Regulator
        if u_id not in self.regulators:
            self.regulators[u_id] = SethDynamicRegulator(copy.deepcopy(self.base_state), self.memory.bge_engine)

        prompt = self.memory.get_full_context(u_id, u_name, text)
        history = self.memory.get_short_term(u_id)
        
        response = await self.engine.ask(prompt, history, text, self.regulators[u_id])
        
        # Persistent Memory Update
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        self.memory.eternal_memory.add(f"{u_name}: {text}. SETH: {response}", user_id=str(u_id), infer=False)
        
        await self._send_long_message(update, response)

    async def _send_long_message(self, update: Update, text: str, max_length: int = 4000):
        """Envía mensajes largos dividiéndolos automáticamente."""
        if not text:
            return

        # Si es corto → envío normal
        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        # Dividir en chunks
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

        # Enviar todos los chunks
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(f"({i+1}/{len(chunks)})\n\n{chunk}")
                
                # Pequeña pausa para no spamear
                await asyncio.sleep(0.3)
                
            except Exception as e:
                logging.error(f"Error sending chunk {i}: {e}")
                # Intento fallback
                await update.message.reply_text("⚠️ La respuesta es muy larga. Aquí va una parte:")
                await update.message.reply_text(chunk[:3500])

    def run(self):
        self.app.run_polling()

if __name__ == '__main__':
    env = SethEnvironment()
    try:
        env.validate()
        bot = SethTelegramBot(
            env=env,
            base_state=SethPresets.DEFAULT,
            engine=SethEngine(env),
            memory=SethMemory(env)
        )
        bot.run()
    except Exception as e:
        logging.error(f"Fatal Bootstrap Error: {e}")
