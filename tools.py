from pathlib import Path
from itertools import islice
from typing import List, Dict, Any

from google.adk.tools import ToolContext
from neo4j_for_adk import tool_success, tool_error
from helper import get_neo4j_import_dir
# =========================
# STATE KEYS
# =========================
PERCEIVED_GOAL_KEY = "perceived_user_goal"
APPROVED_GOAL_KEY = "approved_user_goal"

ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"


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


# =========================
# FILE SELECTION FLOW
# =========================
def set_suggested_files(
    suggested_files: List[str],
    tool_context: ToolContext,
) -> Dict[str, Any]:
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