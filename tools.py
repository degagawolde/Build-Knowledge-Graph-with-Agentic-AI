from pathlib import Path
from itertools import islice
from typing import List, Dict, Any

from google.adk.tools import ToolContext
from neo4j_for_adk import tool_success, tool_error, graphdb
from helper import get_neo4j_import_dir  
from google.adk.tools import ToolContext

# =========================
# STATE KEYS
# =========================
PERCEIVED_GOAL_KEY = "perceived_user_goal"
APPROVED_GOAL_KEY = "approved_user_goal"

ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"

PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"
PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"
SEARCH_RESULTS = "search_results"

# =========================
# HELPERS
# =========================
def get_import_dir() -> Path:
    """Get Neo4j import directory safely."""
    try:
        return Path(get_neo4j_import_dir())
    except Exception as e:
        raise RuntimeError(f"Failed to get import directory: {e}")


def validate_relative_path(file_path: str) -> Path:
    """Ensure file path is relative."""
    path = Path(file_path)
    if path.is_absolute():
        raise ValueError("File path must be relative to import directory.")
    return path


def read_file_sample(full_path: Path, max_lines: int = 100) -> str:
    """Read limited number of lines from file."""
    with open(full_path, "r", encoding="utf-8") as f:
        return "".join(islice(f, max_lines))


# =========================
# GOAL MANAGEMENT
# =========================
def get_approved_user_goal(tool_context: ToolContext):
    goal = tool_context.state.get(APPROVED_GOAL_KEY)
    if not goal:
        return tool_error("No approved goal found.")
    return tool_success(APPROVED_GOAL_KEY, goal)


def set_perceived_user_goal(
    kind_of_graph: str,
    graph_description: str,
    tool_context: ToolContext,
):
    goal_data = {
        "kind_of_graph": kind_of_graph,
        "graph_description": graph_description,
    }
    tool_context.state[PERCEIVED_GOAL_KEY] = goal_data
    return tool_success(PERCEIVED_GOAL_KEY, goal_data)


def approve_perceived_user_goal(tool_context: ToolContext):
    perceived = tool_context.state.get(PERCEIVED_GOAL_KEY)
    if not perceived:
        return tool_error("No perceived goal found.")
    tool_context.state[APPROVED_GOAL_KEY] = perceived
    return tool_success(APPROVED_GOAL_KEY, perceived)


# =========================
# FILE DISCOVERY
# =========================
def list_available_files(tool_context: ToolContext) -> Dict[str, Any]:
    """List all files inside import directory."""
    try:
        import_dir = get_import_dir()
        file_names = [
            str(path.relative_to(import_dir))
            for path in import_dir.rglob("*")
            if path.is_file()
        ]
        tool_context.state[ALL_AVAILABLE_FILES] = file_names
        return tool_success(ALL_AVAILABLE_FILES, file_names)
    except Exception as e:
        return tool_error(f"Failed to list files: {e}")


# =========================
# FILE SAMPLING
# =========================
def sample_file(file_path: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Read up to 100 lines from a file."""
    try:
        relative_path = validate_relative_path(file_path)
        import_dir = get_import_dir()
        full_path = import_dir / relative_path

        if not full_path.exists():
            return tool_error(f"File not found: {file_path}")

        content = read_file_sample(full_path)
        return tool_success("content", content)
    except ValueError as ve:
        return tool_error(str(ve))
    except Exception as e:
        return tool_error(f"Failed to read file: {e}")

# ==========================================
# 🔍 SEARCH & SAMPLING TOOLS
# ==========================================

def search_file(file_path: str, query: str) -> dict:
    """Searches any text file (csv, txt, md) for lines containing the query."""
    import_dir = Path(get_neo4j_import_dir())
    p = import_dir / file_path

    if not p.exists(): return tool_error(f"File does not exist: {file_path}")
    if not query: return tool_success(SEARCH_RESULTS, {"matching_lines": [], "metadata": {"lines_found": 0}})

    matching_lines = []
    try:
        with open(p, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                if query.lower() in line.lower():
                    matching_lines.append({"line_number": i, "content": line.strip()})
    except Exception as e:
        return tool_error(f"Error searching file {file_path}: {e}")

    return tool_success(SEARCH_RESULTS, {
        "metadata": {"path": file_path, "query": query, "lines_found": len(matching_lines)},
        "matching_lines": matching_lines
    })

# =========================
# FILE SELECTION FLOW
# =========================
def set_suggested_files(suggested_files: List[str], tool_context: ToolContext) -> Dict[str, Any]:
    tool_context.state[SUGGESTED_FILES] = suggested_files
    return tool_success(SUGGESTED_FILES, suggested_files)


def get_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    files = tool_context.state.get(SUGGESTED_FILES)
    if not files:
        return tool_error("No suggested files found.")
    return tool_success(SUGGESTED_FILES, files)


def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    suggested = tool_context.state.get(SUGGESTED_FILES)
    if not suggested:
        return tool_error("No suggested files to approve.")
    tool_context.state[APPROVED_FILES] = suggested
    return tool_success(APPROVED_FILES, suggested)


def get_approved_files(tool_context: ToolContext) -> Dict[str, Any]:
    approved = tool_context.state.get(APPROVED_FILES)
    if not approved:
        return tool_error("No approved files found.")
    return tool_success(APPROVED_FILES, approved)


# =========================
# NEO4J DATABASE TOOLS
# =========================
def neo4j_is_ready():
    return graphdb.send_query("RETURN 'Neo4j is Ready!' as message")


def drop_neo4j_indexes() -> Dict[str, Any]:
    """Drops all constraints and indexes present on the neo4j graph database."""
    # Remove constraints
    list_constraints = graphdb.send_query("SHOW CONSTRAINTS YIELD name")
    if list_constraints["status"] == "error":
        return list_constraints
    
    for row in list_constraints["query_result"]:
        name = row["name"]
        graphdb.send_query(f"DROP CONSTRAINT `{name}`")

    # Remove indexes
    list_indexes = graphdb.send_query("SHOW INDEXES YIELD name")
    if list_indexes["status"] == "error":
        return list_indexes
    
    for row in list_indexes["name"]:
        name = row["name"]
        graphdb.send_query(f"DROP INDEX `{name}`")

    return tool_success("message", "Neo4j constraints and indexes have been dropped.")


def clear_neo4j_data() -> Dict[str, Any]:
    """Clears all data from the neo4j graph database."""
    query = "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
    data_removed = graphdb.send_query(query)
    if data_removed["status"] == "error":
        return data_removed
    return tool_success("message", "Neo4j graph has been reset.")


def get_apoc_procedure_names() -> Dict[str, Any]:
    """List all APOC procedure names."""
    cypher = "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc' RETURN name"
    result = graphdb.send_query(cypher)
    if result["status"] == "error":
        return result
    
    names = [row["name"] for row in result["query_result"]]
    if not names:
        return tool_error("APOC procedures not found.")
    return tool_success("apoc_procedure_names", names)


def get_apoc_version() -> Dict[str, Any]:
    """Get the version of APOC installed."""
    result = graphdb.send_query("RETURN apoc.version() AS apoc_version")
    if result["status"] == "error":
        return result
    return tool_success("apoc_version", result["query_result"][0]["apoc_version"])


def get_neo4j_version() -> Dict[str, Any]:
    """Get the version and edition of the Neo4j database."""
    cypher = "CALL dbms.components() yield name, versions, edition unwind versions as version return name, version, edition"
    result = graphdb.send_query(cypher)
    if result["status"] == "error":
        return result
    return tool_success("neo4j_version", result["query_result"][0])


def create_uniqueness_constraint(label: str, unique_property_key: str) -> Dict[str, Any]:
    """Creates a uniqueness constraint."""
    constraint_name = f"{label}_{unique_property_key}_constraint"
    query = f"CREATE CONSTRAINT `{constraint_name}` IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.`{unique_property_key}` IS UNIQUE"
    return graphdb.send_query(query)


def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: list[str],
) -> Dict[str, Any]:
    """Batch loading of nodes from a CSV file."""
    # Fixed formatting for dynamic labels
    query = f"""
    LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    CALL (row) {{
        MERGE (n:`{label}` {{ `{unique_column_name}` : row[$unique_column_name] }})
        FOREACH (k IN $properties | SET n[k] = row[k])
    }} IN TRANSACTIONS OF 1000 ROWS
    """
    return graphdb.send_query(query, {
        "source_file": source_file,
        "unique_column_name": unique_column_name,
        "properties": properties
    })

def load_product_nodes() -> Dict[str, Any]:
    """Load the product nodes from products.csv"""
    return load_nodes_from_csv(
        "products.csv",
        "Product",
        "product_id",
        ["product_name", "price", "description"]
    )


# ==========================================
# 🏗️ STRUCTURED CONSTRUCTION TOOLS
# ==========================================

def propose_node_construction(approved_file: str, proposed_label: str, unique_column_name: str, proposed_properties: list[str], tool_context: ToolContext) -> dict:
    """Propose a node mapping for a structured file."""
    # Safety Check: Does the column exist?
    check = search_file(approved_file, unique_column_name)
    if check["status"] == "error" or check["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} is missing column '{unique_column_name}'. Check file content.")

    plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    plan[proposed_label] = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = plan
    return tool_success("node_construction", plan[proposed_label])

def propose_relationship_construction(approved_file: str, proposed_relationship_type: str, from_node_label: str, from_node_column: str, to_node_label: str, to_node_column: str, proposed_properties: list[str], tool_context: ToolContext) -> dict:
    """Propose a relationship mapping for a structured file."""
    # Safety checks for both foreign key columns
    for col in [from_node_column, to_node_column]:
        check = search_file(approved_file, col)
        if check["status"] == "error" or check["search_results"]["metadata"]["lines_found"] == 0:
            return tool_error(f"{approved_file} missing column '{col}'.")

    plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    plan[proposed_relationship_type] = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = plan
    return tool_success("relationship_construction", plan[proposed_relationship_type])

def approve_proposed_construction_plan(tool_context: ToolContext):
    plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN)
    if not plan: return tool_error("No plan to approve.")
    tool_context.state[APPROVED_CONSTRUCTION_PLAN] = plan
    return tool_success(APPROVED_CONSTRUCTION_PLAN, plan)

# ==========================================
# 📝 UNSTRUCTURED EXTRACTION TOOLS
# ==========================================

def set_proposed_entities(proposed_entity_types: list[str], tool_context: ToolContext):
    tool_context.state[PROPOSED_ENTITIES] = proposed_entity_types
    return tool_success(PROPOSED_ENTITIES, proposed_entity_types)

def approve_proposed_entities(tool_context: ToolContext):
    ents = tool_context.state.get(PROPOSED_ENTITIES)
    if not ents: return tool_error("No entities proposed.")
    tool_context.state[APPROVED_ENTITIES] = ents
    return tool_success(APPROVED_ENTITIES, ents)

def add_proposed_fact(approved_subject_label: str, proposed_predicate_label: str, approved_object_label: str, tool_context: ToolContext):
    approved_ents = tool_context.state.get(APPROVED_ENTITIES, [])
    if approved_subject_label not in approved_ents or approved_object_label not in approved_ents:
        return tool_error("Subject or Object label not in approved entities list.")

    facts = tool_context.state.get(PROPOSED_FACTS, {})
    facts[proposed_predicate_label] = {
        "subject_label": approved_subject_label,
        "predicate_label": proposed_predicate_label,
        "object_label": approved_object_label
    }
    tool_context.state[PROPOSED_FACTS] = facts
    return tool_success(PROPOSED_FACTS, facts)

def approve_proposed_facts(tool_context: ToolContext):
    facts = tool_context.state.get(PROPOSED_FACTS)
    if not facts: return tool_error("No facts to approve.")
    tool_context.state[APPROVED_FACTS] = facts
    return tool_success(APPROVED_FACTS, facts)

def get_well_known_types(tool_context: ToolContext):
    """Bridge tool: pulls node labels from approved construction plan into extraction phase."""
    plan = tool_context.state.get(APPROVED_CONSTRUCTION_PLAN, {})
    labels = {v["label"] for v in plan.values() if v.get("construction_type") == "node"}
    return tool_success("approved_labels", list(labels))

# ==========================================
# 🛠️ UTILITY GETTERS
# ==========================================

def get_proposed_construction_plan(tool_context: ToolContext):
    return tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

def get_approved_entities(tool_context: ToolContext):
    return tool_context.state.get(APPROVED_ENTITIES, [])

def get_proposed_facts(tool_context: ToolContext):
    return tool_context.state.get(PROPOSED_FACTS, {})


def remove_node_construction(label: str, tool_context: ToolContext):
    """Removes a specific node mapping from the proposed plan."""
    plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if label in plan:
        del plan[label]
        tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = plan
        return tool_success("message", f"Node '{label}' removed from plan.")
    return tool_error(f"Node '{label}' not found in plan.")

def remove_relationship_construction(rel_type: str, tool_context: ToolContext):
    """Removes a specific relationship mapping from the proposed plan."""
    plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if rel_type in plan:
        del plan[rel_type]
        tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = plan
        return tool_success("message", f"Relationship '{rel_type}' removed from plan.")
    return tool_error(f"Relationship '{rel_type}' not found in plan.")

def execute_relationship_load(
    source_file: str,
    relationship_type: str,
    from_node_label: str,
    from_node_column: str,
    to_node_label: str,
    to_node_column: str,
    properties: list[str]
) -> Dict[str, Any]:
    """Generic tool to load relationships between existing nodes from CSV."""
    query = f"""
    LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
    MATCH (from:`{from_node_label}` {{ `{from_node_column}`: row[$from_node_column] }})
    MATCH (to:`{to_node_label}` {{ `{to_node_column}`: row[$to_node_column] }})
    MERGE (from)-[r:`{relationship_type}`]->(to)
    FOREACH (k IN $properties | SET r[k] = row[k])
    """
    params = {
        "source_file": source_file,
        "from_node_column": from_node_column,
        "to_node_column": to_node_column,
        "properties": properties
    }
    return graphdb.send_query(query, params)

def finished(tool_context: ToolContext):
    """Signals that the current coordination phase is complete."""
    return tool_success("phase_status", "finished")

def execute_text_to_graph_load(triples: List[Dict[str, str]], tool_context: ToolContext):
    """
    Takes a list of extracted triples and merges them into Neo4j.
    Expected triple format: {'subject': 'James', 'subject_label': 'Person', 'predicate': 'WORKS_AT', 'object': 'Google', 'object_label': 'Company'}
    """
    if not triples:
        return tool_error("No triples provided for loading.")

    query = """
    UNWIND $triples AS triple
    MERGE (s:`{triple.subject_label}` {name: triple.subject})
    MERGE (o:`{triple.object_label}` {name: triple.object})
    MERGE (s)-[r:`{triple.predicate}`]->(o)
    SET r.source = 'NLP_Extraction'
    """
    # Note: Using python f-string carefully or Cypher parameters for labels
    # Since Cypher doesn't allow parameters for Labels, we'll iterate or use a safer APOC method
    
    success_count = 0
    for t in triples:
        cypher = f"""
        MERGE (s:`{t['subject_label']}` {{name: $s_name}})
        MERGE (o:`{t['object_label']}` {{name: $o_name}})
        MERGE (s)-[r:`{t['predicate']}`]->(o)
        RETURN count(r) as loaded
        """
        res = graphdb.send_query(cypher, {"s_name": t['subject'], "o_name": t['object']})
        if res["status"] != "error":
            success_count += 1

    return tool_success("message", f"Successfully loaded {success_count} triples from text.")