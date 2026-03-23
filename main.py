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
logging.basicConfig(level=logging.CRITICAL)
LLM_MODEL = 'gemini-3-flash-preview'

# ==========================================
# 1. CONFIGURATION & FACTORY
# ==========================================
def load_agent_configs(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return {agent['name']: agent for agent in data['agents']}

def create_agent(config, tool_list, name_override=None):
    """Instantiate an Agent object from YAML config."""
    name = name_override or config['name']
    instruction = f"{config['role']}\n\n{config['instructions']}"
    return Agent(
        name=name,
        model=LLM_MODEL,
        description=config.get('description', ''),
        instruction=instruction,
        tools=tool_list
    )

# ==========================================
# 2. SUB-GRAPH BUILDER (LoopAgent Setup)
# ==========================================
def setup_refinement_loop(configs):
    proposal_agent = create_agent(configs['proposal_agent'], [
        tools.sample_file, tools.search_file, 
        tools.propose_node_construction, tools.propose_relationship_construction,
        tools.remove_node_construction, tools.remove_relationship_construction,
        tools.get_approved_user_goal, tools.get_approved_files
    ])

    critic_agent = create_agent(configs['schema_critic_agent'], [
        tools.get_approved_user_goal, tools.get_approved_files,
        tools.get_proposed_construction_plan, tools.sample_file, tools.search_file
    ])

    return LoopAgent(
        name="schema_refinement_loop",
        description="Analyzes approved files to propose a schema based on user intent and feedback.",
        max_iterations=3, 
        sub_agents=[proposal_agent, critic_agent]
    )

# ==========================================
# 3. INTERACTIVE ENGINE
# ==========================================
async def run_interactive_phase(caller, phase_name, completion_key):
    print(f"\n{'='*40}\n PHASE: {phase_name} \n{'='*40}")
    
    while True:
        user_input = input("👤 You: ").strip()
        if not user_input: continue
        if user_input.lower() in ['exit', 'quit']: return False
        
        await caller.call(user_input)
        
        session = await caller.get_session()
        if completion_key in session.state:
            print(f"\n✅ {phase_name} complete!")
            return True

# ==========================================
# 4. MAIN ORCHESTRATION PIPELINE
# ==========================================
async def main():
    load_env()
    configs = load_agent_configs('agents.yml')
    shared_state = {}

    # --- PHASE 1: USER INTENT ---
    intent_tools = [tools.set_perceived_user_goal, tools.approve_perceived_user_goal]
    intent_agent = create_agent(configs['user_intent_agent'], intent_tools)
    intent_caller = await make_agent_caller(intent_agent, initial_state=shared_state)
    if not await run_interactive_phase(intent_caller, "USER INTENT", "approved_user_goal"): return
    shared_state = (await intent_caller.get_session()).state

    # --- PHASE 2: FILE SELECTION ---
    file_tools = [tools.get_approved_user_goal, tools.list_available_files, tools.sample_file, 
                  tools.set_suggested_files, tools.get_suggested_files, tools.approve_suggested_files]
    file_agent = create_agent(configs['file_suggestion_agent'], file_tools)
    file_caller = await make_agent_caller(file_agent, initial_state=shared_state)
    await file_caller.call("Start by listing available files and suggesting the best ones.")
    if not await run_interactive_phase(file_caller, "FILE SELECTION", "approved_files"): return
    shared_state = (await file_caller.get_session()).state

    # --- ROUTING LOGIC ---
    approved_files = shared_state.get("approved_files", [])
    has_structured = any(f.endswith(('.csv', '.json', '.xlsx')) for f in approved_files)
    has_unstructured = any(f.endswith(('.txt', '.md')) for f in approved_files)

    # --- PHASE 3: STRUCTURED SCHEMA PROPOSAL ---
    if has_structured:
        print("\n📊 Starting Coordinated Schema Proposal...")
        refinement_loop = setup_refinement_loop(configs)
        refinement_loop_tool = agent_tool.AgentTool(refinement_loop)

        coordinator_tools = [
            refinement_loop_tool,
            tools.get_proposed_construction_plan,
            tools.approve_proposed_construction_plan,
            tools.get_approved_user_goal,
            tools.get_approved_files,
            tools.finished
        ]
        coordinator_agent = create_agent(configs['schema_proposal_coordinator'], coordinator_tools)
        coord_caller = await make_agent_caller(coordinator_agent, initial_state=shared_state)
        
        await coord_caller.call("How can these approved files be imported into our graph?")
        if not await run_interactive_phase(coord_caller, "STRUCTURED SCHEMA COORDINATION", "approved_construction_plan"): return
        shared_state = (await coord_caller.get_session()).state

    # --- PHASE 4: UNSTRUCTURED (NER & FACTS) ---
    if has_unstructured:
        print("\n📝 Detected Unstructured Data. Starting Extraction Schema Design...")
        
        # 4a. NER Schema
        ner_tools = [tools.get_approved_user_goal, tools.get_approved_files, tools.sample_file, 
                     tools.get_well_known_types, tools.set_proposed_entities, tools.approve_proposed_entities]
        ner_agent = create_agent(configs['ner_schema_agent_v1'], ner_tools)
        ner_caller = await make_agent_caller(ner_agent, initial_state=shared_state)
        await ner_caller.call("Analyze the text files and propose entity types that match our goal.")
        if not await run_interactive_phase(ner_caller, "UNSTRUCTURED: NER", "approved_entity_types"): return
        shared_state = (await ner_caller.get_session()).state

        # 4b. Fact Type Schema
        fact_tools = [tools.get_approved_user_goal, tools.get_approved_files, tools.get_approved_entities,
                      tools.sample_file, tools.add_proposed_fact, tools.get_proposed_facts, tools.approve_proposed_facts]
        fact_agent = create_agent(configs['fact_type_extraction_agent_v1'], fact_tools)
        fact_caller = await make_agent_caller(fact_agent, initial_state=shared_state)
        await fact_caller.call("Identify relevant relationship types (facts) between our entities.")
        if not await run_interactive_phase(fact_caller, "UNSTRUCTURED: FACTS", "approved_fact_types"): return
        shared_state = (await fact_caller.get_session()).state

        # 4c. NLP Extraction Execution
        print("\n🧠 Running NLP Extraction Worker...")
        extract_tools = [tools.get_approved_entities, tools.get_approved_facts, tools.get_approved_files, 
                         tools.sample_file, tools.execute_text_to_graph_load]
        extract_agent = create_agent(configs['text_extraction_worker'], extract_tools)
        extract_caller = await make_agent_caller(extract_agent, initial_state=shared_state)
        await extract_caller.call("Extract instances of entities and facts from text files and load them into Neo4j.")
        if not await run_interactive_phase(extract_caller, "UNSTRUCTURED: NLP EXTRACTION", "extraction_complete"): return
        shared_state = (await extract_caller.get_session()).state

    # --- PHASE 5: FINAL CONSTRUCTION (STRUCTURED LOAD) ---
    if "approved_construction_plan" in shared_state:
        print("\n Building Structured Knowledge Graph in Neo4j...")
        build_tools = [tools.execute_node_load, tools.execute_relationship_load, tools.neo4j_is_ready, tools.clear_neo4j_data]
        build_agent = create_agent(configs['graph_builder_agent'], build_tools)
        build_caller = await make_agent_caller(build_agent, initial_state=shared_state)
        await build_caller.call("The structured schema is approved. Execute the data migration now.")
        await run_interactive_phase(build_caller, "STRUCTURED MIGRATION", "migration_complete")

    print("\n🚀 All processing branches complete. Knowledge Graph is live.")

if __name__ == "__main__":
    asyncio.run(main())