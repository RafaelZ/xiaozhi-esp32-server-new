# Refactoring Plan: Migration to LangGraph

## 1. Analysis of Existing Architecture
The current `xiaozhi-server` uses a monolithic `ConnectionHandler` class to manage conversation state, ASR/TTS streams, and LLM interactions.
- **Pros**: Low latency optimization for audio streams; explicit control flow.
- **Cons**: High coupling; difficult to extend (e.g., adding risk assessment checks requires modifying the core loop); state management is fragile (manual flags).

## 2. Proposed Architecture (LangGraph)
We will refactor the core agent logic into a LangChain `StateGraph`.

### Core Components
1.  **State (`AgentState`)**:
    - `messages`: List of conversation messages.
    - `risk_level`: Current risk assessment (Low, High).
    - `user_profile`: User context (summary of past days).
2.  **Nodes**:
    - **`RiskGuard`**: Analyzes user input for psychological risks before generating a response.
    - **`PsychIntervention`**: Specialized node for high-risk scenarios.
    - **`MainAgent`**: Standard conversation flow (handling tools, chit-chat).
    - **`ToolExecutor`**: Executes external tools.
3.  **Edges**:
    - `RiskGuard` -> `PsychIntervention` (if risk is High)
    - `RiskGuard` -> `MainAgent` (if risk is Low)
    - `MainAgent` -> `ToolExecutor` (if tool call detected)
    - `ToolExecutor` -> `MainAgent`

### Integration
- **Audio/WebSocket Layer**: Remains in `ConnectionHandler` to handle VAD and streaming.
- **Logic Layer**: `ConnectionHandler` will invoke the compiled LangGraph runnable with the user's text input.

## 3. New Features
### A. Real-time Risk Assessment
- Implemented as the first node in the graph (`RiskGuard`).
- Uses a lightweight prompt or fast model to classify input.
- High risk triggers an immediate transition to the `PsychIntervention` node.

### B. Periodic Summary (1:00 AM)
- **Mechanism**: An external scheduler (e.g., `APScheduler` or a simple background loop) triggers a summary job.
- **Logic**:
    1.  Load conversation history from the persistence layer (Postgres/SQLite via LangGraph checkpointer).
    2.  Run a "Summarization Agent" to analyze mood and key events.
    3.  Save the summary to `UserProfile` to be loaded into the `AgentState` for the next session.

## 4. Implementation Steps
1.  **Dependencies**: Add `langchain`, `langgraph`.
2.  **Graph Implementation**: Create `core/agent/graph.py` containing the nodes and graph definition.
3.  **Refactor ConnectionHandler**: Replace the `chat()` method's internal logic with `app.invoke()` or `app.stream()` from the graph.
4.  **Scheduler**: Add the background job for daily summaries.
