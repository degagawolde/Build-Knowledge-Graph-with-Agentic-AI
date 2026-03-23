This script is a sophisticated **multi-agent orchestration pipeline**. Its goal is to take raw data (files like CSVs, JSONs, or Text) and transform them into a **Neo4j Knowledge Graph** by using a series of specialized AI agents.

Think of it as a factory line where each "station" is an AI agent that handles one specific part of the data engineering process.

---

## 1. High-Level Architecture
The script follows a linear "Phase" approach. It uses a **Shared State** (a dictionary passed between agents) to ensure that the decisions made by the first agent (e.g., "What is the user's goal?") are known by the last agent (e.g., "How do I write the database query?").



### The Core Components
* **YAML Configs:** Instead of hardcoding prompts, it loads agent definitions (roles and instructions) from an `agents.yml` file.
* **`LoopAgent` (The Refinement Loop):** This is a unique ADK feature. It creates a mini-ecosystem where a **Proposal Agent** and a **Critic Agent** argue back and forth up to 3 times to perfect a database schema before showing it to you.
* **The Interactive Engine:** The `run_interactive_phase` function keeps the script in a `while` loop, allowing you to chat with the current agent until a specific "completion key" (like `approved_files`) appears in the session state.

---

## 2. The Five Phases of the Pipeline

### Phase 1: User Intent 🎯
The `user_intent_agent` talks to you to figure out what you are trying to build. 
* **Goal:** Define a "Perceived User Goal."
* **Exit Condition:** When you approve the goal, `approved_user_goal` is saved to the state.

### Phase 2: File Selection 📂
The `file_suggestion_agent` looks at your local folder.
* **Action:** It lists files and suggests which ones are relevant to your goal.
* **Exit Condition:** You approve the list of files (`approved_files`).

### Phase 3 & 4: The Branching Logic (Structured vs. Unstructured)
The script checks the file extensions of your approved files:
* **If Structured (CSV/JSON/XLSX):** It triggers the **Refinement Loop**. A coordinator oversees the creation of a graph schema (Nodes and Relationships) based on the table headers.
* **If Unstructured (TXT/MD):** It triggers an **NLP Pipeline**. 
    1.  **NER Agent:** Identifies "Entity" types (e.g., Person, Organization).
    2.  **Fact Agent:** Identifies "Relationship" types (e.g., WORKS_AT).
    3.  **Extraction Worker:** Actually reads the text and pulls out specific data points.

### Phase 5: Final Construction 🏗️
Once the schemas are approved and the data is extracted, the `graph_builder_agent` takes over.
* **Action:** It uses Neo4j tools to actually run the `LOAD CSV` or `CREATE` commands to populate your database.

---

## 3. Key Technical Patterns

### State-Driven Control Flow
The script uses the agent's **Session State** as a logic gate.
```python
if completion_key in session.state:
    return True # Moves to the next Phase
```
This is a robust way to handle AI agents because it ensures the pipeline only moves forward when the AI has successfully called a specific tool (like `approve_suggested_files`).

### The "Coordinator" Pattern
In Phase 3, the script uses a `schema_proposal_coordinator`. This agent doesn't do the heavy lifting itself; it has the `LoopAgent` as a **tool**. It "hires" the loop to do the work and then presents the results to the user.



---

## Summary of the Workflow

| Phase | Agent Role | Output |
| :--- | :--- | :--- |
| **1. Intent** | Understand what the user wants. | `approved_user_goal` |
| **2. Files** | Pick the right data sources. | `approved_files` |
| **3. Schema** | Design the Graph (Nodes/Edges). | `approved_construction_plan` |
| **4. NLP** | Extract facts from plain text. | `extraction_complete` |
| **5. Build** | Write data into Neo4j. | `migration_complete` |