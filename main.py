# main.py
import asyncio
import yaml
import logging
import warnings
from google.adk.agents import Agent, LoopAgent
from google.adk.tools import agent_tool

# Local Module Imports
from helper import make_agent_caller, load_env
import tools

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)  # Reduced noise
LLM_MODEL = 'gemini-3-flash-preview'

# ==========================================
# 1. CORE UTILITIES
# ==========================================
def load_agent_configs(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return {agent['name']: agent for agent in data['agents']}

def create_agent(config, tool_list, name_override=None):
    name = name_override or config['name']
    instruction = f"{config['role']}\n\n{config['instructions']}"
    return Agent(
        name=name, model=LLM_MODEL,
        description=config.get('description', ''),
        instruction=instruction, tools=tool_list
    )

# ==========================================
# 2. INTERACTIVE ENGINE (WITH BACKTRACKING)
# ==========================================
async def run_phase(caller, phase_name, completion_key):
    """
    Handles user interaction.
    Returns:
      - True: Phase complete
      - False: User wants to exit
      - "BACK": User wants to return to previous phase
    """
    print(f"\n{'='*40}\n ENTERING PHASE: {phase_name} \n{'='*40}")
    print(" (Type 'back' to go to previous step, 'exit' to quit)")

    while True:
        user_input = input(f"👤 [{phase_name}] You: ").strip()
        if not user_input:
            continue

        low_input = user_input.lower()
        if low_input in ['exit', 'quit']:
            return False
        if low_input == 'back':
            return "BACK"

        try:
            await caller.call(user_input)
            session = await caller.get_session()
            if completion_key in session.state:
                print(f"✅ {phase_name} verified.")
                return True
        except Exception as e:
            print(f"❌ Error in {phase_name}: {e}")

# ==========================================
# 3. PHASE-SPECIFIC LOGIC
# ==========================================

async def process_structured_data(configs, shared_state):
    """Handles CSV/JSON to Graph transformation."""
    print("\n📊 Starting Structured Data Pipeline...")

    # Setup LoopAgent (Actor-Critic)
    proposal_agent = create_agent(configs['proposal_agent'], [
        tools.get_approved_user_goal,
        tools.get_approved_files,
        tools.sample_file,
        tools.search_file,
        tools.propose_node_construction,
        tools.propose_relationship_construction,
        tools.get_proposed_construction_plan
    ])
    critic_agent = create_agent(configs['schema_critic_agent'], [
        tools.get_approved_user_goal,
        tools.get_approved_files,
        tools.get_proposed_construction_plan,
        tools.sample_file,
        tools.search_file
    ])
    refinement_loop = LoopAgent(
        name="refinement_loop", max_iterations=3,
        sub_agents=[proposal_agent, critic_agent]
    )

    coordinator_agent = create_agent(configs['schema_proposal_coordinator'], [
        agent_tool.AgentTool(refinement_loop),
        tools.get_approved_user_goal,
        tools.get_proposed_construction_plan,
        tools.approve_proposed_construction_plan,
        tools.remove_node_construction, 
        tools.remove_relationship_construction,
        tools.finished
    ])

    caller = await make_agent_caller(coordinator_agent, initial_state=shared_state)
    await caller.call("Generate a graph schema for these files.")
    result = await run_phase(caller, "STRUCTURED SCHEMA", "approved_construction_plan")

    if result is True:
        # Final Construction
        build_agent = create_agent(configs['graph_builder_agent'], [
            tools.get_approved_construction_plan,
            tools.execute_node_load,
            tools.execute_relationship_load,
            tools.drop_neo4j_schema,
            tools.clear_database,
            tools.neo4j_is_ready
        ])
        build_caller = await make_agent_caller(build_agent, initial_state=(await caller.get_session()).state)
        await build_caller.call("Execute the data migration now.")
        await run_phase(build_caller, "MIGRATION", "migration_complete")

# main.py additions

async def process_unstructured_data(configs, shared_state):
    """Handles NER and NLP extraction via sequential Agent coordination."""
    print("\n📝 Starting NLP Pipeline...")

    # 1. Entity Recognition Phase
    ner_agent = create_agent(configs['ner_schema_agent_v1'], [
        tools.get_well_known_types,
        tools.get_approved_user_goal,
        tools.get_approved_files,
        tools.sample_file,
        tools.set_proposed_entities,
        tools.approve_proposed_entities
    ])
    ner_caller = await make_agent_caller(ner_agent, initial_state=shared_state)
    await ner_caller.call("Analyze the text files and propose entity types.")
    if not await run_phase(ner_caller, "ENTITY DISCOVERY", "approved_entity_types"):
        return

    # 2. Fact/Relationship Extraction Phase
    shared_state = (await ner_caller.get_session()).state
    fact_agent = create_agent(configs['fact_type_extraction_agent_v1'], [
        tools.get_approved_user_goal,
        tools.get_approved_files,
        tools.sample_file,
        tools.get_proposed_facts,
        tools.get_approved_entities,
        tools.add_proposed_fact,
        tools.approve_proposed_facts
    ])
    fact_caller = await make_agent_caller(fact_agent, initial_state=shared_state)
    await fact_caller.call("Based on the approved entities, what relationships exist in the text?")
    if not await run_phase(fact_caller, "FACT DISCOVERY", "approved_fact_types"):
        return

    # 3. Final Execution
    worker_agent = create_agent(configs['text_extraction_worker'], [
        tools.execute_text_to_graph_load,
        tools.get_approved_facts,
        tools.get_approved_files,
        tools.sample_file
    ])
    worker_caller = await make_agent_caller(worker_agent, initial_state=(await fact_caller.get_session()).state)
    await worker_caller.call("Extract the triples and load them into Neo4j.")

# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================
async def main():
    load_env()
    configs = load_agent_configs('agents.yml')
    shared_state = {}
    history = []  # Stack for backtracking

    # List of phases to iterate through
    # Format: (Display Name, State Key, Agent Config, Tools)
    pipeline_definition = [
        ("USER INTENT", "approved_user_goal", 'user_intent_agent',
         [
             tools.set_perceived_user_goal, 
             tools.approve_perceived_user_goal]),

        ("FILE SELECTION", "approved_files", 'file_suggestion_agent',
         [
             tools.get_approved_user_goal,
             tools.list_available_files, 
             tools.sample_file, 
             tools.set_suggested_files, 
             tools.get_suggested_files,
             tools.approve_suggested_files
             ])
    ]

    current_idx = 0
    while current_idx < len(pipeline_definition):
        name, key, cfg_key, t_list = pipeline_definition[current_idx]

        agent = create_agent(configs[cfg_key], t_list)
        caller = await make_agent_caller(agent, initial_state=shared_state)

        result = await run_phase(caller, name, key)

        if result == "BACK":
            current_idx = max(0, current_idx - 1)
            continue
        elif result is False:
            print("👋 Exiting...")
            return

        # Advance
        shared_state = (await caller.get_session()).state
        current_idx += 1

    # --- PARALLEL EXECUTION BRANCH ---
    approved_files = shared_state.get("approved_files", [])
    tasks = []

    if any(f.endswith(('.csv', '.json')) for f in approved_files):
        tasks.append(process_structured_data(configs, shared_state))

    if any(f.endswith(('.txt', '.md')) for f in approved_files):
        tasks.append(process_unstructured_data(configs, shared_state))

    if tasks:
        await asyncio.gather(*tasks)

    print("\n🚀 Knowledge Graph Deployment Complete.")

if __name__ == "__main__":
    asyncio.run(main())