import asyncio
import yaml
import logging
import warnings
from google.adk.agents import Agent

# Local Module Imports
from helper import make_agent_caller
import tools 

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
llm = 'gemini-3-flash-preview'

# ==========================================
# 1. CONFIGURATION & FACTORY
# ==========================================
def load_agent_configs(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return {agent['name']: agent for agent in data['agents']}

def create_agent(config, tool_list):
    """Instantiate an Agent object from YAML config."""
    instruction = f"{config['role']}\n\n{config['instructions']}"
    return Agent(
        name=config['name'],
        model=llm,
        description=config['description'],
        instruction=instruction,
        tools=tool_list
    )

# ==========================================
# 2. INTERACTIVE CHAT ENGINE
# ==========================================
async def run_interactive_phase(caller, phase_name, completion_key):
    """
    Runs a loop allowing the user to chat with the agent.
    Exits once the 'completion_key' is found in the session state.
    """
    print(f"\n{'='*20}")
    print(f" PHASE: {phase_name} ")
    print(f"{'='*20}")
    print(f"(Type 'status' to see current state or 'exit' to force quit)\n")

    while True:
        user_input = input("👤 You: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() == 'exit':
            return False
        if user_input.lower() == 'status':
            session = await caller.get_session()
            print(f"DEBUG State: {session.state}")
            continue

        # Send input to agent
        await caller.call(user_input)
        
        # Check if the phase goal was achieved
        session = await caller.get_session()
        if completion_key in session.state:
            print(f"\n✅ {phase_name} complete! Moving to next phase...")
            return True

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
async def main():
    # Load YAML Profiles
    configs = load_agent_configs('agents.yml')
    shared_state = {}

    # --- PHASE 1: USER INTENT ---
    intent_tools = [tools.set_perceived_user_goal, tools.approve_perceived_user_goal]
    intent_agent = create_agent(configs['user_intent_agent'], intent_tools)
    intent_caller = await make_agent_caller(intent_agent, initial_state=shared_state)
    
    # Start the interactive loop for Intent
    success = await run_interactive_phase(intent_caller, "USER INTENT", "approved_user_goal")
    if not success: return

    # Sync state for the next agent
    shared_state = (await intent_caller.get_session()).state

    # --- PHASE 2: FILE SUGGESTION ---
    file_tools = [
        tools.get_approved_user_goal, 
        tools.list_available_files, 
        tools.sample_file, 
        tools.set_suggested_files, 
        tools.get_suggested_files,
        tools.approve_suggested_files
    ]
    file_agent = create_agent(configs['file_suggestion_agent'], file_tools)
    file_caller = await make_agent_caller(file_agent, initial_state=shared_state)
    
    # Nudge the agent to start by listing files automatically
    await file_caller.call("List the available files and suggest which ones I should use for my goal.")
    
    success = await run_interactive_phase(file_caller, "FILE SELECTION", "approved_files")
    if not success: return

    # Sync state for the next agent
    shared_state = (await file_caller.get_session()).state

    # --- PHASE 3: NER PROPOSAL ---
    ner_tools = [
        tools.get_approved_user_goal,
        tools.get_approved_files,
        tools.sample_file
    ]
    ner_agent = create_agent(configs['ner_agent'], ner_tools)
    ner_caller = await make_agent_caller(ner_agent, initial_state=shared_state)
    
    # Nudge the agent to start the analysis
    await ner_caller.call("Analyze the approved files and propose the entity types for our graph.")
    
    # Since NER is usually the final step or requires its own approval tool:
    await run_interactive_phase(ner_caller, "ENTITY EXTRACTION", "proposed_entities")

    print("\n🚀 All phases complete. Your Knowledge Graph plan is ready.")

if __name__ == "__main__":
    asyncio.run(main())