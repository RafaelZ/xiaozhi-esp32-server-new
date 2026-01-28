# Xiaozhi Server Refactor Analysis & Plan

## 1. Project Analysis

The current `xiaozhi-server` is a sophisticated, asyncio-based backend for ESP32 AI hardware. The core logic resides mainly in `main/xiaozhi-server/core/connection.py`.

### Core Architecture
*   **Entry Point:** `app.py` initializes a `WebSocketServer`.
*   **Session Management:** `ConnectionHandler` is the monolithic class managing the entire lifecycle of a user session. It handles WebSocket frames, audio buffering, VAD (Voice Activity Detection), ASR (Speech-to-Text), and TTS (Text-to-Speech).
*   **Agent Logic:** The "Agent" is implemented as a method `chat()` within `ConnectionHandler`.
    *   **Pattern:** It uses a manual recursive "ReAct" (Reasoning + Acting) loop.
    *   **Execution:** `chat(depth=0)` calls the LLM. If the LLM requests tool calls (detected via specific string patterns or API-native function calls), it executes them via `UnifiedToolHandler` and recursively calls `chat(depth=n+1)`.
    *   **State:** State is scattered across instance attributes of `ConnectionHandler` (`self.dialogue`, `self.client_is_speaking`, `self.memory`, etc.).
*   **Memory:** Managed by `MemoryProviderBase` with implementations for local file-based summaries (`mem_local_short`) and external APIs (`mem0ai`). It injects history into the system prompt.
*   **Routing:** `TextMessageHandlerRegistry` routes JSON messages based on a `type` field.

### Current "Agent" Implementation (Python)
The `chat` method in `connection.py` manually handles:
1.  Constructing the message history (`self.dialogue.get_llm_dialogue_with_memory`).
2.  Calling the LLM (`self.llm.response_with_functions`).
3.  Parsing the response for tool calls (handling both native function calls and text-based JSON).
4.  Executing tools via `self.func_handler`.
5.  Recursively calling itself with the tool outputs.
6.  Managing depth limits (`MAX_DEPTH = 5`).

## 2. Comparison with LangChain 1.0+ (LangGraph)

LangChain 1.0 introduced **LangGraph** as the standard for building stateful, multi-actor applications (Agents). This is a paradigm shift from the manual recursion seen in `ConnectionHandler`.

| Feature | Current Implementation | LangChain 1.0+ (LangGraph) |
| :--- | :--- | :--- |
| **Control Flow** | **Manual Recursion:** `chat()` calls `chat()` inside `if tool_call:` blocks. Harder to visualize and debug complex flows. | **Cyclic Graph:** Nodes (functions) and Edges (conditions) defined explicitly. The flow is a state machine. |
| **State Management** | **Implicit/Object-Oriented:** State is held in `self` attributes of `ConnectionHandler`. | **Explicit/Schema-Based:** A shared `State` TypedDict is passed between nodes. Immutable-style updates. |
| **Tool Calling** | **Custom Logic:** Manual parsing of JSON strings or API responses. `UnifiedToolHandler` wraps execution. | **Standardized:** `bind_tools` attaches tools to LLMs. `ToolNode` automatically executes them. |
| **Memory/Persistence** | **Custom Provider:** `MemoryProvider` saves to YAML/API. History injected manually. | **Checkpointers:** Built-in persistence layer (Sqlite/Postgres) that saves the `State` automatically at every step. |
| **Streaming** | **Manual:** Iterating over generator `llm_responses` and handling chunks. | **Built-in:** `.stream()` and `.astream_events()` provide standardized token/event streaming. |
| **Observability** | **Logging:** Standard Python logging. | **LangSmith:** First-class integration for tracing every step, token usage, and latency. |

### Pros & Cons of Migration

**LangChain 1.0+ (LangGraph) Pros:**
*   **Structure:** De-couples the "Agent" logic from the "Connection" infrastructure.
*   **Flexibility:** Adding a "Risk Check" or "Human in the Loop" is just adding a Node/Edge, not modifying a recursive function.
*   **Standardization:** Uses community-standard interfaces for Tools and LLMs.
*   **Persistence:** `checkpointer` makes it trivial to pause/resume conversations or handle long-running background tasks (like the daily summary).

**LangChain 1.0+ Cons:**
*   **Complexity:** Introduces a learning curve and new abstractions.
*   **Overhead:** Might add slight latency compared to raw API calls (though negligible for LLMs).
*   **Streaming Integration:** Hooking LangGraph's stream events into the existing WebSocket audio pipeline requires careful bridging.

## 3. Feasibility Analysis

**Feasibility:** **High**.
The current logic is a standard "Tool Calling Agent". LangGraph's `create_react_agent` or a prebuilt `ToolNode` graph maps 1:1 to this pattern. The `UnifiedToolHandler` can be adapted to expose LangChain-compatible `BaseTool` instances.

**Risks:**
*   **Real-time Latency:** The "Risk Check" feature implies an extra model call. If not optimized, this increases response time.
*   **Audio Pipeline:** The current `ConnectionHandler` is tightly coupled with ASR/TTS streams. The new Agent must yield text chunks immediately for TTS, which LangGraph supports but requires correct implementation (`astream`).

## 4. Rewrite Plan & Architecture

### New Feature Integration
1.  **Real-time Psychological Risk Detection:**
    *   **Architecture:** A **conditional parallel node** or a **pre-processing node** in the graph.
    *   **Implementation:** Before the main "Agent" node, a lightweight `RiskGuard` node analyzes the user's latest input.
        *   *Option A (Fast):* Regex/Keyword matching + small BERT model.
        *   *Option B (Accurate):* A fast LLM call (e.g., GPT-3.5/Haiku) with a specific prompt: "Is this user expressing high psychological risk? Yes/No".
    *   **Flow:** If `High Risk` -> Route to `InterventionNode` (empathetic script, alert system) -> End. If `Low Risk` -> Route to `AgentNode`.

2.  **Daily Summary (1 AM):**
    *   **Architecture:** An external scheduler (e.g., `APScheduler` or System Cron) triggers a separate Graph.
    *   **Implementation:**
        *   LangGraph Checkpointer stores all session history in a DB (SQLite/Postgres).
        *   At 1 AM, the job queries the DB for the user's history from the last 24h.
        *   A `SummarizerGraph` processes this text and updates a `UserProfile` document.
        *   The main Agent Graph loads this `UserProfile` into the context for the next day.

### Proposed Architecture (LangGraph)

The monolithic `ConnectionHandler` will be refactored. The `chat()` method will be replaced by a `LangGraphRunner`.

**Graph State (`AgentState`):**
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_profile: str  # From Daily Summary
    risk_level: str
```

**Nodes:**
1.  **`LoadContext`**: Fetches long-term memory/summary.
2.  **`RiskGuard`**: (New Feature) Analyzes input for self-harm/depression signals.
3.  **`Agent`**: The LLM model bound with tools.
4.  **`Tools`**: Executes the tools (IoT control, Weather, etc.).
5.  **`SaveContext`**: (Optional) Explicit memory saving, though Checkpointer handles state.

**Edges:**
*   `Start` -> `LoadContext` -> `RiskGuard`
*   `RiskGuard` --(High Risk)--> `Intervention` -> `End`
*   `RiskGuard` --(Safe)--> `Agent`
*   `Agent` --(Call Tool)--> `Tools` -> `Agent`
*   `Agent` --(Final Answer)--> `End`

### Migration Steps

1.  **Tool Adaptation:** Wrap existing `plugins_func` into LangChain `BaseTool` or `@tool` decorators.
2.  **Graph Construction:** Define the `StateGraph` as described above.
3.  **Integration:**
    *   In `ConnectionHandler`, initialize the Graph runnable.
    *   Replace the `chat()` loop with `await graph.astream(inputs)`.
    *   Map the output stream chunks to the existing `tts.tts_text_queue`.
4.  **Risk Node Implementation:** Add the specialized prompt/model for risk detection.
5.  **Scheduler:** Add a background thread/process for the 1 AM summary task.

### Difficulties & Solutions
*   **Difficulty:** Integrating the existing `UnifiedToolHandler` which uses custom dynamic loading.
    *   *Solution:* Write a bridge class that iterates `UnifiedToolHandler.get_functions()` and dynamically creates `StructuredTool` objects for LangChain.
*   **Difficulty:** "Intervention" must stop the normal flow.
    *   *Solution:* LangGraph's conditional edges perfect for this. If risk is detected, the graph routes to a static response node and skips the generic LLM.

### Dependencies
*   `langchain>=0.2`
*   `langgraph>=0.1`
*   `langchain-openai` (or community providers)
*   `pydantic`
