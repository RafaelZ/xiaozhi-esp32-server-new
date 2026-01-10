# XiaoZhi Agent Refactor Analysis & Proposal

This document provides a comprehensive analysis of the current XiaoZhi agent architecture, compares it with a modern LangChain 1.0+ (LangGraph) approach, and proposes a detailed plan for refactoring, including the integration of new psychological risk features.

## 1. Project Analysis (Current State)

### 1.1 Core Architecture
The current implementation of `xiaozhi-server` relies on a **monolithic "God Object" pattern**. The central component is `ConnectionHandler` (in `core/connection.py`), which manages the entire lifecycle of a user session.

*   **State Management**: State is maintained via instance variables (`self.client_is_speaking`, `self.dialogue`, `self.sentence_id`, etc.) within the `ConnectionHandler`.
*   **Event Loop**: A manual event loop combines `asyncio` for non-blocking I/O (WebSocket, LLM streaming) and `threading` for blocking operations (some audio processing, reporting).
*   **Message Routing**: `_route_message` manually dispatches text and audio data. Audio flows into a queue (`asr_audio_queue`) processed by VAD and ASR modules.
*   **Tool Execution**: Custom implementation (`UnifiedToolHandler`) that parses raw string outputs from LLMs to detect and execute function calls.
*   **Memory**: Integrated with `mem0ai` via manual hooks (`_save_and_close`, `query_memory`).

### 1.2 Observations
*   **High Coupling**: Network logic (WebSocket), Business logic (Chat), and Audio processing are tightly coupled in one class.
*   **Fragility**: Adding new steps (e.g., a "Risk Check" before response) requires modifying the complex `chat` method, increasing the risk of regressions.
*   **Concurrency**: The mix of threads and async tasks makes debugging race conditions difficult.

---

## 2. Comparison with LangChain 1.0+ (LangGraph)

LangChain 1.0 introduced **LangGraph**, a framework specifically designed for building stateful, multi-actor applications with LLMs.

| Feature | Current Implementation | LangChain 1.0+ (LangGraph) |
| :--- | :--- | :--- |
| **Control Flow** | Imperative `if/else` logic inside `chat()` and `_route_message`. | **Declarative Graph**: Nodes (functions) and Edges (logic) define the flow. |
| **State Management** | Ad-hoc instance variables (`self.foo`). | **StateSchema**: A typed dictionary (Pydantic/TypedDict) passed between nodes. |
| **Tool Calling** | Custom regex/JSON parsing and execution loop. | **Native Support**: `bind_tools` and `ToolNode` handle parsing, validation, and execution automatically. |
| **Streaming** | Manual generator handling and WebSocket pushes. | **`astream_events`**: Standardized event streaming API (tokens, tool calls, state updates). |
| **Persistence** | Manual `save_memory` hooks. | **Checkpointers**: Built-in state persistence (Postgres, SQLite) supporting "time-travel" and resuming sessions. |
| **Observability** | Standard logging (`logger.info`). | **LangSmith**: Native deep tracing of every step, latency, and token usage. |

### Pros & Cons of Refactoring
*   **Pros**:
    *   **Modularity**: Logic is broken into small, testable nodes (e.g., `RiskCheckNode`, `ResponseNode`).
    *   **Maintainability**: The workflow is visible and easier to modify.
    *   **Ecosystem**: Instant access to 100+ integrations without writing custom wrappers.
    *   **New Features**: Adding "Psychological Risk Check" becomes just inserting one node into the graph.
*   **Cons**:
    *   **Latency**: LangChain introduces a slight overhead (milliseconds). For real-time voice, this is critical, but LangGraph is optimized for this.
    *   **Complexity**: Shift from imperative to graph-based thinking.

---

## 3. Feasibility Study

**Verdict: Highly Feasible.**

Rewriting with LangChain 1.0+ is not only feasible but recommended for the long-term health of the project, especially given the new "Psychological Risk" requirements which imply complex conditional logic.

### 3.1 Addressing Real-time Requirements
*   **Streaming**: LangGraph's `.astream_events()` fits perfectly with the WebSocket requirement, allowing us to stream tokens to the TTS engine as they are generated.
*   **Latency**: By using `async` nodes, the overhead is minimal. The ASR and TTS components can remain as specialized services/modules invoked by the graph nodes.

### 3.2 Benefits for New Requirements
1.  **Real-time Risk Identification**: Can be implemented as a **conditional edge** or a **parallel node** in the graph. If "Risk" is detected, the graph routes to an `InterventionNode` instead of the standard `ChatNode`.
2.  **Periodic Summarization**: LangGraph's **Checkpointer** architecture saves the state of every interaction to a database. A separate background job (CRON) can simply load the state from the DB and run a `SummaryGraph` without interfering with the live session.

---

## 4. Rewrite Proposal & Architecture

### 4.1 New Architecture: The "Mind" Graph
We will replace the logic inside `ConnectionHandler` with a `compiled` LangGraph.

#### The State Schema
```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_profile: dict  # Loaded from DB
    risk_level: str     # 'low', 'medium', 'high'
    current_mood: str
    audio_buffer: bytes # Temporary storage for current turn
```

#### The Graph Nodes
1.  **`AudioInputNode`**: (Kept external to graph usually, or as entry) Receives text from ASR.
2.  **`RiskAssessmentNode`**:
    *   *Input*: User's latest message + recent history.
    *   *Logic*: Uses a specialized fast model (e.g., fine-tuned BERT or minimal LLM prompt) to classify risk.
    *   *Output*: Updates `risk_level`.
3.  **`Router` (Conditional Edge)**:
    *   If `risk_level == 'high'`: Go to `InterventionNode`.
    *   Else: Go to `MainAgentNode`.
4.  **`InterventionNode`**:
    *   Generates a supportive, de-escalating response.
    *   Notifies human operators (if configured).
5.  **`MainAgentNode`**:
    *   Standard LLM processing with Tools.
6.  **`ToolNode`**:
    *   Executes registered tools.
7.  **`TTSOutputNode`**: (Usually a callback or post-processing) streaming audio back.

### 4.2 Implementation Strategy for New Features

#### Feature 1: Real-time Psychological Risk Identification
*   **Model**: To ensure low latency (<200ms), we should not use a full generic LLM for this if possible.
    *   *Option A (Recommended)*: Use a specialized, small Classification Model (e.g., DistilBERT fine-tuned on sentiment/suicide-risk datasets) running locally via ONNX.
    *   *Option B*: Use a very fast LLM API (e.g., Groq Llama3-8b or GPT-3.5-Turbo) with a strict JSON prompt.
*   **Integration**: This runs in parallel with fetching context. If high risk is flagged, it aborts the standard generation.

#### Feature 2: Periodic Summarization (1 AM Daily)
*   **Mechanism**:
    1.  Use **PostgresCheckpointer** (or SQLite) for LangGraph persistence.
    2.  Write a separate script `daily_summary.py`.
    3.  Script runs via CRON at 01:00.
    4.  Logic:
        *   Query all unique `thread_id`s (users) active in the last 24h.
        *   Load their state: `graph.get_state(config)`.
        *   Run a `SummaryChain` on the day's messages.
        *   Update the `user_profile` in the state or save to a `DailyReport` table.

### 4.3 Proposed Directory Structure
```
main/xiaozhi-server/
└── core/
    ├── agent/              # NEW: LangGraph Logic
    │   ├── graph.py        # Main graph definition
    │   ├── nodes/          # Individual nodes
    │   │   ├── risk.py     # Risk assessment logic
    │   │   ├── chat.py     # LLM wrapping
    │   │   └── tools.py    # ToolNode configuration
    │   └── state.py        # TypedDict schema
    ├── connection.py       # Refactored: thinner, wraps the Graph
    └── ...
```

## 5. Next Steps
1.  **Environment**: Upgrade `requirements.txt` to include `langchain`, `langgraph`, `langchain-openai` (or relevant provider).
2.  **Prototype**: Create `core/agent/graph.py` implementing the basic Chat + Risk flow.
3.  **Integration**: Modify `connection.py` to instantiate the Graph runner instead of the manual `chat()` logic.
