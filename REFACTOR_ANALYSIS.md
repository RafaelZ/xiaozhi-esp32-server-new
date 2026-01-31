# Xiaozhi-Server Refactor Analysis & Plan

## 1. Codebase Analysis (Current Implementation)

The `xiaozhi-server` is a Python-based backend for the Xiaozhi AI hardware, primarily handling WebSocket connections for real-time interaction.

### Core Components
*   **Entry Point:** `app.py` initializes `WebSocketServer`.
*   **Session Management:** `core/connection.py` defines `ConnectionHandler`. A new instance is created for *each* WebSocket connection, maintaining the state (`_vad`, `_asr`, `_llm`, `_memory`, `dialogue`, etc.) for that session.
*   **Event Loop:**
    *   The `handle_connection` method in `ConnectionHandler` runs an async loop reading messages.
    *   Audio data is routed to `_process_websocket_audio` -> `asr_audio_queue`.
    *   Text messages are routed to `handleTextMessage` -> `TextMessageProcessor`.
*   **Agent Logic (The "Brain"):**
    *   Implemented in `ConnectionHandler.chat()`.
    *   **Mechanism:** It uses a **manual recursive loop** to handle Multi-turn Tool Use (ReAct pattern).
    *   **Flow:**
        1.  User Query -> Memory Query.
        2.  LLM Call (`response` or `response_with_functions`).
        3.  If `tool_calls` detected -> Execute tools via `UnifiedToolHandler`.
        4.  Recursively call `chat()` with tool outputs (incrementing `depth`).
        5.  Stop when `depth > MAX_DEPTH` or no tool calls.
*   **Tools:** managed by `UnifiedToolHandler` and `ToolManager`. Supports Server Plugins, MCP (Model Context Protocol), and IoT device control.
*   **Memory:** Abstracted via `MemoryProviderBase`. Implementations include `mem0ai` and `mem_local_short`.

### Weaknesses of Current Architecture
1.  **Hard-coded Control Flow:** The agentic loop is hard-coded in `chat()` with `if/else` and recursion. Adding complex logic (e.g., "Human in the loop", "Parallel execution", "Conditional branching based on sentiment") requires modifying the core recursive function, which is error-prone.
2.  **State Management:** State is scattered across `ConnectionHandler` instance attributes. There is no clear, unified state schema (like `GraphState` in LangGraph), making debugging and state serialization difficult.
3.  **Observability:** While logging exists (`config.logger`), there is no built-in tracing for the chain of thought (CoT) or tool execution steps comparable to LangSmith.
4.  **Scalability of Logic:** As features grow (e.g., adding the requested "Psychological Risk Detection"), the `chat` function will become a monolithic "God Object".

---

## 2. Comparison: Current vs. LangChain >1.0 (LangGraph)

| Feature | Current Implementation | LangChain >1.0 (LangGraph) |
| :--- | :--- | :--- |
| **Orchestration** | Manual Recursion & Loops | **Graph-based (Nodes & Edges)** |
| **State** | Instance Attributes (`self.x`) | **TypedDict Schema** (Immutable-style updates) |
| **Tool Calling** | Manual parsing & execution | **Pre-built ToolNode** & `bind_tools` |
| **Memory** | Custom `MemoryProvider` | **Checkpointers** (Thread-level persistence) |
| **Streaming** | Custom generator handling | Standardized `.stream()` / `astream_events` |
| **Flexibility** | Low (Hard to change flow) | **High** (Add/Remove nodes easily) |
| **New Features** | Requires invasive code changes | Add new Nodes (`RiskCheck`, `Summary`) |

### Pros & Cons of Refactoring

**Pros:**
*   **Modularity:** Logic is broken down into small, testable Nodes.
*   **Visualizability:** The graph structure clearly represents the agent's logic.
*   **Extensibility:** Easy to insert the "Psychological Risk" node without breaking existing chat logic.
*   **Ecosystem:** Access to LangChain's vast integration library (Retrievers, Tools, Models).
*   **Future-Proofing:** Standardized interface for Agents.

**Cons:**
*   **Rewrite Effort:** Significant. `ConnectionHandler` is deeply coupled with the WebSocket logic.
*   **Latency:** LangChain introduces a slight overhead compared to raw API calls (negligible for this use case).
*   **Complexity:** Learning curve for LangGraph concepts (State, Reducers).

---

## 3. Feasibility Study

**Verdict:** **Highly Feasible and Recommended.**
The current project structure separates the *Transport Layer* (WebSocket/Audio) from the *Reasoning Layer* (`chat`). We can replace the `chat` method's internal logic with a **compiled LangGraph Runnable**, while keeping the existing WebSocket handling and Audio processing pipeline intact.

### Benefits for New Requirements
1.  **Real-time Psychological Risk:** A `RiskCheckNode` can be placed *before* the main Agent Node. It can short-circuit the flow to an `InterventionNode` if high risk is detected, ensuring immediate response.
2.  **Daily Summary:** Can be implemented as a separate graph or a background task that reuses the same LangChain components (LLM, Memory) but with a different prompt/chain.

---

## 4. Rewrite Plan & Architecture (LangChain 1.0+)

### Architecture Blueprint

The `ConnectionHandler` will no longer manage the loop. It will instantiate a `LangGraph` agent and simply stream inputs to it.

#### A. State Schema
```python
from typing import TypedDict, Annotated, List, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    user_id: str
    risk_level: str  # "normal", "monitor", "high"
    sentiment_score: float
```

#### B. The Graph (Nodes)
1.  **`guardrail_node` (New):**
    *   **Input:** User message.
    *   **Action:** Uses a lightweight model or specialized prompt to classify `risk_level`.
    *   **Output:** Updates `risk_level` in state.
2.  **`agent_node`:**
    *   **Action:** The main Persona (Xiaozhi). It decides whether to chat or call tools.
    *   **Condition:** If `risk_level == "high"`, it switches system prompt to "Crisis Intervention Mode".
3.  **`tools_node`:**
    *   **Action:** Executes tools (Weather, IoT, Memory Query).
    *   **Integration:** Wrap existing `UnifiedToolHandler` tools as `LangChain` Tools.
4.  **`summary_node` (Scheduled):**
    *   **Action:** Runs daily at 1 AM. Reads past 24h messages, generates a summary, and saves to `Memory`.

#### C. Control Flow (Edges)
*   `START` -> `guardrail_node`
*   `guardrail_node` --(high risk)--> `agent_node` (with Intervention Prompt)
*   `guardrail_node` --(normal)--> `agent_node` (Standard Persona)
*   `agent_node` --(tool_calls)--> `tools_node`
*   `tools_node` -> `agent_node`
*   `agent_node` -> `END`

### Implementation Roadmap

#### Phase 1: Infrastructure Setup
1.  Add `langchain`, `langgraph`, `langchain-openai` (or generic) to `requirements.txt`.
2.  Create `core/agent_graph/` directory.

#### Phase 2: Tool Wrapping
1.  Create an adapter to convert `UnifiedToolHandler` tools into `LangChain` compatible tools (`BaseTool`).

#### Phase 3: Graph Construction
1.  Implement `RiskAssessment` chain.
2.  Implement the Main Agent Graph using `StateGraph`.
3.  Replace the body of `ConnectionHandler.chat()` to invoke `self.agent_graph.ainvoke(...)`.

#### Phase 4: Periodic Tasks (Daily Summary)
1.  Integrate `APScheduler` in `app.py`.
2.  Create a job that queries the `MemoryProvider` for all active users, runs a "Summary Chain", and saves the result back.

### Key Difficulties & Risks
*   **Streaming Compatibility:** The current system relies heavily on streaming tokens to TTS. LangGraph's `.stream()` outputs events (chunks). We must adapt the `ConnectionHandler` to consume LangGraph events and push them to `tts.tts_text_queue`.
*   **State Persistence:** LangGraph uses Checkpointers. We need to decide whether to use LangGraph's built-in persistence (e.g., Postgres/Sqlite) or sync it with the existing `MemoryProvider`. *Recommendation: Keep existing MemoryProvider for long-term storage, use LangGraph Memory only for conversation turns.*
