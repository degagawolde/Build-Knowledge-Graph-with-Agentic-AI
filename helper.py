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
# 1. CONFIGURATION & ENVIRONMENT
# ==========================================

# Configure logging for a professional "startup" feel
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ADK-Helper")

_ENV_LOADED = False

def load_env():
    global _ENV_LOADED
    if not _ENV_LOADED:
        env_file = find_dotenv()
        load_dotenv(env_file)
        _ENV_LOADED = True


def get_env_var(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """Fetch environment variable with optional strict enforcement."""
    load_env()
    val = os.getenv(key, default)
    if required and not val:
        raise ValueError(f"Missing mandatory environment variable: {key}")
    return val

def get_openai_api_key():
    return get_env_var("OPENAI_API_KEY")

def get_neo4j_import_dir() -> Path:
    """Returns the Neo4j import directory as a resolved Path object."""
    path_str = get_env_var("NEO4J_IMPORT_DIR")
    path = Path(path_str).resolve()
    if not path.exists():
        logger.error(f"Neo4j Import Directory does not exist: {path}")
    return path

def sanitize_name(name: str) -> str:
    """Sanitizes labels/types to prevent Cypher injection."""
    return re.sub(r'[^a-zA-Z0-9_]', '', name)
# ==========================================
# 2. AGENT CALLER (Orchestration Wrapper)
# ==========================================

class AgentCaller:
    """
    A refined wrapper for ADK agents that handles event streams,
    state persistence, and error logging.
    """
    
    def __init__(self, agent: Agent, runner: Runner, user_id: str, session_id: str):
        self.agent = agent
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id
        self.logger = logging.getLogger(f"Agent:{agent.name}")

    async def get_session(self):
        """Fetch the current state of the agent's session."""
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name, 
            user_id=self.user_id, 
            session_id=self.session_id
        )

    async def call(self, query: str, verbose: bool = False) -> str:
        """
        Executes a turn with the agent and returns the final text response.
        Handles tool calls and multi-turn logic automatically via the ADK Runner.
        """
        self.logger.info(f"Query: {query}")

        content = types.Content(role='user', parts=[types.Part(text=query)])
        final_response_text = ""

        try:
            async for event in self.runner.run_async(
                user_id=self.user_id, 
                session_id=self.session_id, 
                new_message=content
            ):
                if verbose:
                    self.logger.debug(f"Event: {type(event).__name__} | Author: {event.author}")

                # Check for the concluding turn response
                if event.is_final_response():
                    if event.content and event.content.parts:
                        # Extract text from the first available part
                        final_response_text = next(
                            (p.text for p in event.content.parts if p.text), 
                            "Agent produced an empty response."
                        )
                    
                    # Handle failures or escalations
                    if event.actions and event.actions.escalate:
                        error_msg = event.error_message or "Unknown escalation"
                        self.logger.error(f"Agent Escalated: {error_msg}")
                        final_response_text = f"Error: {error_msg}"
                    
                    # Break only if the responding agent is our target
                    if event.author == self.agent.name:
                        break

        except Exception as e:
            self.logger.exception("Critcal error during agent execution:")
            return f"System Error: {str(e)}"

        self.logger.info(f"Response: {final_response_text[:100]}...")
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
    session_service = InMemorySessionService()
    app_name = f"{agent.name}_app"
    user_id = "system_user"
    
    # Initialize the session with the provided state
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