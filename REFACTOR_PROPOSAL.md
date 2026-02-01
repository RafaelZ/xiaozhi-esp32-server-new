# Refactoring Analysis & Proposal: Migrating Xiaozhi-Server to LangChain 1.0+

## 1. Current Implementation Analysis

The current `xiaozhi-server` backend relies on a **custom-built, monolithic agent architecture** primarily located in `core/connection.py`.

### Key Characteristics:
*   **Manual ReAct Loop:** The `chat()` method in `ConnectionHandler` implements a recursive function to handle the Observation-Thought-Action loop. It manually manages recursion depth (`depth` param) to prevent infinite loops.
*   **State Management:**
    *   **Runtime State:** Held in `self.dialogue` (in-memory list of `Message` objects) and various instance variables (`client_audio_buffer`, `sentence_id`).
    *   **Persistence:** Abbreviated memory saving via `MemoryProviderBase` (e.g., `mem_local_short`), but the runtime state is ephemeral to the WebSocket connection.
*   **Tooling:** A complex, custom `UnifiedToolHandler` and `ToolManager` system (`core/providers/tools/`) handles disparate tool types (IoT, MCP, Plugins). It manually constructs OpenAI-compatible function schemas.
*   **Coupling:** The agent logic is tightly coupled with I/O handlers. VAD (Voice Activity Detection), ASR (Speech-to-Text), and TTS (Text-to-Speech) triggers are interleaved within the connection logic.

### Pros & Cons of Current State:
*   **Pros:** Highly optimized for low latency (crucial for voice); complete control over every byte of the stream; no framework overhead.
*   **Cons:** Hard to maintain; "Spaghetti code" in `connection.py` (mix of network logic, business logic, and agent logic); difficult to add complex flows (like multi-agent collaboration or complex guardrails).

---

## 2. Comparison with LangChain 1.0+ (LangGraph)

LangChain 1.0 introduces a stable API, and **LangGraph** is the de-facto standard for building agentic workflows in the ecosystem.

| Feature | Current Implementation | LangChain 1.0+ / LangGraph |
| :--- | :--- | :--- |
| **Control Flow** | Recursive Python function (`chat` calling `chat`). Hardcoded `if/else` logic. | **StateGraph:** A graph-based finite state machine. Nodes represent steps (LLM, Tool, etc.), Edges represent transitions. |
| **Memory/State** | `Dialogue` class (list wrapper) + custom persistence. | **Checkpointers:** Built-in state persistence (Postgres, SQLite, Memory). Allows "Time Travel" and resuming sessions. |
| **Tool Calling** | Manual parsing of `tool_calls` from LLM response & manual execution loop. | **Prebuilt ToolNode:** Standardized execution. `bind_tools` method for LLMs. |
| **Streaming** | Custom async generators handling text chunks & TTS triggers. | **.stream() / .astream_events():** Standardized event streaming (tokens, tool calls, state updates). |
| **Guardrails** | Embedded `if` checks (e.g., depth limit). | **Conditional Edges:** Logic to route flow (e.g., `if risk > high then goto InterventionNode`). |

---

## 3. Feasibility & Risk Assessment

### Feasibility: **High**
The core logic of `xiaozhi-server` (receive text -> process -> tool/LLM -> output) maps directly to a graph structure. Python 3.10 is supported.

### Benefits:
1.  **Decoupling:** Business logic (Risk, Intent, Chat) is separated from Transport logic (WebSocket, MQTT).
2.  **Observability:** LangSmith integration comes for free (vital for debugging agent loops).
3.  **Extensibility:** Adding the requested "Psychological Risk Detection" becomes a simple node insertion, rather than hacking the `chat` function.
4.  **Ecosystem:** Access to thousands of community tools and retrievers (RAG).

### Risks:
1.  **Latency:** LangChain adds a slight overhead. For a real-time voice assistant, every millisecond counts. *Mitigation: Use `LCEL` (LangChain Expression Language) for tight loops and keep the graph simple.*
2.  **Migration Effort:** `ConnectionHandler` is massive. Rewriting it requires ripping out the brain while keeping the nervous system (ASR/TTS) intact.
3.  **Dependency Weight:** Adding `langchain`, `langgraph`, `pydantic` increases docker image size.

---

## 4. Proposed Architecture (LangChain 1.0 / LangGraph)

We propose wrapping the agent logic into a **LangGraph StateGraph**. The `ConnectionHandler` will act as the "Runner" for this graph, feeding it inputs and consuming events to trigger TTS.

### 4.1 Core Graph Structure

**State Schema:**
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    risk_level: str
    sentiment: str
```

**Nodes:**
1.  **RiskGuard (New Feature):** Analyzes the latest input for psychological risk.
    *   *Implementation:* Lightweight LLM call or Classification Model.
2.  **Agent (LLM):** The core conversational model. Decides to call tools or respond.
3.  **Tools:** Executes requested tools (IoT, Weather, etc.).
4.  **Intervention (New Feature):** A specialized prompt path if `RiskGuard` detects high risk.

**Flow:**
`Start` -> `RiskGuard` -> (Conditional Edge)
   *   If `High Risk` -> `Intervention` -> `End`
   *   If `Normal` -> `Agent` -> (Conditional Edge)
       *   If `ToolCall` -> `Tools` -> `Agent`
       *   If `Response` -> `End`

### 4.2 Handling Specific Requirements

#### Requirement 1: Real-time Psychological Risk Intervention
*   **Solution:** The **RiskGuard Node**.
*   **Mechanism:** Before the main LLM sees the message, the RiskGuard classifies it.
*   **Edge Logic:**
    ```python
    def route_risk(state):
        if state['risk_level'] == 'CRITICAL':
            return "intervention_node"
        return "agent_node"
    ```
*   **Benefit:** Prevents the general chatbot from joking about serious topics. Immediate, specialized handling.

#### Requirement 2: Periodic (Daily 1 AM) User Summary
*   **Solution:** **Async Background Worker + Shared Checkpointer**.
*   **Architecture:**
    1.  Use a persistent Checkpointer (e.g., `AsyncSqliteSaver` or `PostgresSaver`) for the main conversation.
    2.  Create a separate background script/process (controlled by `APScheduler` or simple Cron).
    3.  At 1 AM, this script loads the state for all active users using the shared Checkpointer.
    4.  It runs a "Summarization Graph": `Load History` -> `Summarize LLM` -> `Save to Profile/DB`.
*   **Benefit:** Does not impact the runtime latency of the voice chat. Uses the standard history storage.

### 4.3 Refactor Roadmap

1.  **Phase 1: Hybrid Approach.** Install LangChain. Create a `LangChainAgent` class. In `connection.py`, replace the manual `llm.response` call with `LangChainAgent.ainvoke()`. Keep tools legacy for now.
2.  **Phase 2: Tool Migration.** Wrap existing `UnifiedToolHandler` tools into `LangChain` tools (`@tool` decorator or `StructuredTool`).
3.  **Phase 3: LangGraph Implementation.** Move the control flow (recursion, risk check) into a compiled `StateGraph`.
4.  **Phase 4: Persistence.** Switch from `mem_local_short` to LangGraph Checkpointers.

### 4.4 Example Code Snippet (Conceptual)

```python
# graph_builder.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    risk_score: float

async def risk_analyzer(state):
    # Fast, specialized model check
    score = await analyze_risk(state["messages"][-1])
    return {"risk_score": score}

async def chatbot(state):
    return {"messages": [llm.invoke(state["messages"])]}

async def intervention(state):
    return {"messages": [AIMessage(content="I detect you are going through a hard time...")]}

def route_risk(state):
    if state["risk_score"] > 0.8:
        return "intervention"
    return "chatbot"

workflow = StateGraph(State)
workflow.add_node("risk_analyzer", risk_analyzer)
workflow.add_node("chatbot", chatbot)
workflow.add_node("intervention", intervention)

workflow.set_entry_point("risk_analyzer")
workflow.add_conditional_edges("risk_analyzer", route_risk)
workflow.add_edge("intervention", END)
workflow.add_edge("chatbot", END)

app = workflow.compile()
```
