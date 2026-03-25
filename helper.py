import os
import logging
from pathlib import Path
import re
from typing import Optional, Dict, Any, Union

from dotenv import load_dotenv, find_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================

# Create a logs directory locally
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_system.log"

# Define a professional format with a clear timestamp
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),   # Persistent local file
        logging.StreamHandler()          # Console output for real-time tracking
    ]
)

logger = logging.getLogger("ADK-Helper")
logger.info(f"--- SYSTEM INITIALIZED | Logging to {LOG_FILE} ---")

_ENV_LOADED = False

def load_env():
    global _ENV_LOADED
    if not _ENV_LOADED:
        env_file = find_dotenv()
        load_dotenv(env_file)
        _ENV_LOADED = True

def get_env_var(key: str, default: Optional[str] = None, required: bool = True) -> str:
    load_env()
    val = os.getenv(key, default)
    if required and not val:
        logger.error(f"CRITICAL: Missing environment variable {key}")
        raise ValueError(f"Missing mandatory environment variable: {key}")
    return val

def get_neo4j_import_dir() -> Path:
    path_str = get_env_var("NEO4J_IMPORT_DIR")
    path = Path(path_str).resolve()
    if not path.exists():
        logger.error(f"Neo4j Import Directory does not exist: {path}")
    return path

def sanitize_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '', name)

# ==========================================
# 2. AGENT CALLER (Orchestration Wrapper)
# ==========================================

class AgentCaller:
    """
    A refined wrapper for ADK agents that handles event streams,
    state persistence, and detailed local file logging.
    """
    
    def __init__(self, agent: Agent, runner: Runner, user_id: str, session_id: str):
        self.agent = agent
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        # Named logger based on the specific agent
        self.logger = logging.getLogger(f"Agent:{agent.name}")

    async def get_session(self):
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, 
            user_id=self.user_id, 
            session_id=self.session_id
        )

    async def call(self, query: str, verbose: bool = False) -> str:
        """
        Executes a turn and logs critical steps to the local file.
        """
        # CRITICAL STEP: Input Capture
        self.logger.info(f"▶️  START TURN | Session: {self.session_id} | Query: '{query}'")

        content = types.Content(role='user', parts=[types.Part(text=query)])
        final_response_text = ""

        try:
            async for event in self.runner.run_async(
                user_id=self.user_id, 
                session_id=self.session_id, 
                new_message=content
            ):
                if verbose:
                    self.logger.info(f"  ∟ [EVENT] {type(event).__name__} from {event.author}")

                # CRITICAL STEP: Identify the final response
                if event.is_final_response():
                    self.logger.info(f"✅ FINAL RESPONSE RECEIVED from {event.author}")
                    
                    if event.content and event.content.parts:
                        final_response_text = next(
                            (p.text for p in event.content.parts if p.text), 
                            "Agent produced an empty response."
                        )
                    
                    # CRITICAL STEP: Handle failures/escalations
                    if event.actions and event.actions.escalate:
                        error_msg = event.error_message or "Unknown escalation"
                        self.logger.warning(f"⚠️  AGENT ESCALATED: {error_msg}")
                        final_response_text = f"Error: {error_msg}"
                    
                    if event.author == self.agent.name:
                        break

        except Exception as e:
            # CRITICAL STEP: System Failure
            self.logger.error(f"❌ CRITICAL ERROR during agent execution: {str(e)}", exc_info=True)
            return f"System Error: {str(e)}"

        # CRITICAL STEP: Completion Summary
        self.logger.info(f"🏁 TURN COMPLETED | Response snippet: {final_response_text[:50]}...")
        return final_response_text

# ==========================================
# 3. FACTORY METHODS
# ==========================================

async def make_agent_caller(
    agent: Agent, 
    initial_state: Optional[Dict[str, Any]] = None,
    session_id: str = "default_session"
) -> AgentCaller:
    """
    Bootstrap a new AgentCaller with an isolated in-memory session.
    """
    logger.info(f"🛠️  BOOTSTRAP: Initializing agent caller for '{agent.name}'")
    
    session_service = InMemorySessionService()
    app_name = f"{agent.name}_app"
    user_id = "system_user"
    
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state or {}
    )
    
    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )
    
    return AgentCaller(agent, runner, user_id, session_id)