# Agent Refactor Plan: Migration to LangChain 1.0+ (LangGraph)

## 1. Current Architecture Analysis

The current "XiaoZhi" agent is implemented primarily in Python using a custom, monolithic architecture.

### Core Components
*   **ConnectionHandler (`core/connection.py`)**: This is the "God Object". It manages:
    *   **WebSocket Lifecycle**: Connection, message routing (`_route_message`), and closure.
    *   **State Management**: Holds `self.dialogue` (conversation history) and `self.memory`.
    *   **Orchestration Loop**: The `chat()` method implements a manual "ReAct" style loop. It calls the LLM, parses the output for function calls (`_merge_tool_calls`), executes tools (`UnifiedToolHandler`), and recursively calls itself (`chat(depth=depth+1)`).
    *   **I/O Handling**: Directly manages VAD (Voice Activity Detection), ASR (Speech-to-Text), and TTS (Text-to-Speech) queues.
*   **Dialogue (`core/utils/dialogue.py`)**: A custom class managing a list of `Message` objects. It handles prompt engineering (injecting system prompts, memories, and voiceprint info).
*   **UnifiedToolHandler**: A custom abstraction for loading and executing local Python functions as tools.

### Observations
*   **Imperative Logic**: The control flow is hardcoded in Python methods. Adding complex logic (e.g., "if risk is high, branch here") increases the complexity of the already massive `ConnectionHandler`.
*   **Manual Tool Parsing**: The agent manually parses JSON from LLM outputs to detect tool calls, which is error-prone compared to native function calling support in modern SDKs.
*   **Tight Coupling**: The agent logic is tightly coupled with the WebSocket connection and audio processing.

---

## 2. Comparison: Current vs. LangChain 1.0 (LangGraph)

LangChain 1.0 introduced **LangGraph**, a library for building stateful, multi-actor applications with LLMs. It models agent workflows as graphs (nodes and edges).

| Feature | Current Implementation | LangChain 1.0+ (LangGraph) |
| :--- | :--- | :--- |
| **Control Flow** | Imperative (recursive `chat` function). Hard to visualize or debug complex flows. | Declarative (Graph of Nodes/Edges). Visualizable, modular, and explicit. |
| **State Management** | Custom `Dialogue` class held in memory (RAM). Lost on restart unless manually saved. | `StateSchema` (Pydantic/TypedDict). Built-in Persistence (Checkpointers) allows resuming state across server restarts seamlessly. |
| **Tool Calling** | Manual JSON parsing and execution loop. | Standardized `ToolNode`. Supports parallel tool calls and streaming out-of-the-box. |
| **Observability** | Logging to files/console. Hard to trace chains. | Native integration with **LangSmith** for tracing, debugging, and evaluating every step. |
| **Streaming** | Custom logic handling chunks and TTS generation. | Built-in streaming events (`on_chat_model_stream`, etc.). |
| **Ecosystem** | Limited to manually implemented providers. | Access to hundreds of integrations (Vector DBs, Retrievers, 3rd party APIs). |

### Pros & Cons of Migration

**Pros:**
*   **Maintainability**: Decouples the "Agent Logic" from the "Connection/Audio Logic".
*   **Scalability**: Easier to add new flows (e.g., "Risk Intervention Flow") without breaking the main loop.
*   **Reliability**: Relies on battle-tested libraries for tool parsing and state management.
*   **Features**: Free access to memory, persistence, and time-travel debugging.

**Cons:**
*   **Latency**: LangChain adds a slight overhead compared to raw API calls (negligible for this use case).
*   **Complexity**: Learning curve for LangGraph concepts (State, Nodes, Edges).
*   **Migration Effort**: Requires rewriting the core `chat` logic.

---

## 3. Feasibility & Risk

**Feasibility:** **High**.
The current logic (Receive -> LLM -> Tool -> Response) maps perfectly to a simple `StateGraph`. The Python environment (3.10) is compatible.

**Risk:** **Medium**.
*   **Audio Sync**: The current system tightly coordinates text generation with TTS. Migrating to LangChain requires carefully handling the streaming output to ensure TTS starts immediately.
*   **Prompt Regression**: Porting the custom prompt construction (`Dialogue` class) to LangChain prompts requires testing to ensure character consistency.

---

## 4. Refactor Proposal: LangGraph Architecture

We will replace the manual `chat()` loop in `ConnectionHandler` with a compiled `LangGraph` Runnable.

### A. Graph State Schema

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    risk_level: str  # "normal", "low", "high", "critical"
    session_id: str
```

### B. The Graph Structure (Nodes & Edges)

1.  **`Input Node`**: Accepts user text (from ASR).
2.  **`RiskGuard Node` (New Feature)**:
    *   **Function**: Analyzes the user's latest input + context for psychological risk.
    *   **Model**: A fast, specialized prompt (or smaller model like `gpt-3.5-turbo` or a fine-tuned local model) to classify risk.
    *   **Output**: Updates `risk_level` in state.
3.  **`Router Edge`**:
    *   If `risk_level` is "high/critical" -> Go to `InterventionNode`.
    *   Else -> Go to `AgentNode`.
4.  **`InterventionNode`**:
    *   **Function**: Generates a supportive, de-escalating response using a specific "Therapist Persona" prompt. Skips standard tools to avoid distractions.
5.  **`AgentNode` (Standard Flow)**:
    *   **Function**: The main LLM (XiaoZhi) that decides to call tools or chat.
    *   **Model**: Main LLM (e.g., GPT-4o, DeepSeek).
6.  **`ToolNode`**:
    *   **Function**: Executes selected tools (Weather, IoT, Memory).
7.  **`ResponseGenerator`**:
    *   **Function**: Finalizes the text for TTS.

### C. Psychological Risk Implementation (Detail)

To minimize latency, the `RiskGuard` can run in parallel with the `AgentNode` (using LangGraph's parallel execution support), but for safety, it's better as a **pre-check**.

*   **Prompt Strategy**:
    ```text
    Analyze the following user input for signs of suicide risk, self-harm, or severe emotional distress.
    Return ONLY a JSON: {"risk_level": "high" | "normal", "reason": "..."}
    User: {input}
    ```
*   **Action**: If "high", the graph diverts to a flow that notifies admins (via webhook/log) and prompts the LLM to enter "Crisis Intervention Mode".

---

## 5. Periodic Summary Implementation (New Feature)

**Requirement**: "Analyze user psychological changes daily at 1 AM."

**Architecture**:
Since `ConnectionHandler` processes are tied to active WebSocket connections, we need a **background scheduler** (independent of the user being online).

1.  **Storage**: Ensure LangGraph uses a persistent checkpointer (e.g., `AsyncSqliteSaver` or `PostgresSaver`). This saves all conversation threads.
2.  **Scheduler**: Use a library like `APScheduler` in `app.py` or a separate `cron` service.
3.  **The Summary Agent**:
    *   Triggers at 1 AM.
    *   **Step 1**: Query the database for all `threads` active in the last 24 hours.
    *   **Step 2**: For each user, load their conversation history.
    *   **Step 3**: Invoke a "Summarization Chain" (LLM).
        *   *Prompt*: "Review the user's conversations from today. Identify mood shifts, key stressors, and psychological state. Update the 'Psychological Profile'."
    *   **Step 4**: Save the new profile to the `LongTermMemory` (or User Profile DB).

### Integration Plan

1.  **Install Dependencies**: `langchain`, `langgraph`, `langchain-openai`, `langchain-community`.
2.  **Define Graph**: Create `core/agent_graph.py`.
3.  **Modify `ConnectionHandler`**:
    *   Remove `self.chat()`.
    *   Initialize `self.graph_app` on connection.
    *   In `_route_message`, call `await self.graph_app.ainvoke(inputs)`.
    *   Stream outputs to `self.tts`.

This approach modernizes the stack, enables the required psychological features in a modular way, and significantly improves maintainability.
