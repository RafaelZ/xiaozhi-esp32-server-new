import asyncio
from typing import Literal, Dict, Any, cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from core.agent.state import AgentState

# --- Node Definitions ---

async def agent_node(state: AgentState, config: RunnableConfig):
    """
    The main agent node responsible for generating responses.
    It considers the risk level and user profile.
    """
    # Retrieve configuration options
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gpt-4o-mini")
    # API Key is automatically handled by langchain-openai if env var is set
    # or we can pass api_key=configurable.get("api_key") if needed.

    llm = ChatOpenAI(model=model_name, temperature=0, streaming=True)

    # Retrieve tools
    tools = configurable.get("tools", [])

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    # Construct System Prompt based on Profile and Risk
    system_prompt = "You are a helpful voice assistant named Xiaozhi."

    if state.get("user_profile"):
        profile = state["user_profile"]
        system_prompt += f"\n\nUser Profile: {profile}"

    if state.get("risk_level") == "high":
        system_prompt += "\n\nWARNING: High psychological risk detected. Be empathetic, supportive, and suggest professional help if needed. Do not be dismissive."

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # Invoke LLM (Streaming is handled by the caller using astream_events)
    response = await llm_with_tools.ainvoke(messages, config)

    return {"messages": [response]}


async def risk_detection_node(state: AgentState, config: RunnableConfig):
    """
    Parallel node to detect psychological risk in the user's latest message.
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return {"risk_level": "low"}

    text = last_message.content.lower()

    # Heuristic / Mock Logic for Risk Detection
    high_risk_keywords = ["suicide", "kill myself", "end my life", "want to die"]
    medium_risk_keywords = ["depressed", "sad", "lonely", "hopeless"]

    if any(k in text for k in high_risk_keywords):
        return {"risk_level": "high"}
    elif any(k in text for k in medium_risk_keywords):
        return {"risk_level": "medium"}

    return {"risk_level": "low"}


def route_tools(state: AgentState) -> Literal["tools", "__end__"]:
    """
    Determine whether to continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "__end__"


def intervention_gate(state: AgentState) -> Literal["agent", "intervention"]:
    """
    Gate to decide if we proceed with normal agent response or intervention.
    """
    if state.get("risk_level") == "high":
        return "intervention"
    return "agent"

async def intervention_node(state: AgentState, config: RunnableConfig):
    """
    Node to handle high-risk situations.
    """
    # This response effectively "interrupts" the normal flow.
    # In a real system, we might trigger a specific TTS output directly or log an alert.
    msg = AIMessage(content="I hear that you are going through a difficult time. I am an AI, but there are people who can help. Please consider calling a support hotline.")
    return {"messages": [msg]}

# --- Graph Construction ---

def build_agent_graph(tools=None):
    if tools is None:
        tools = []

    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("risk_detector", risk_detection_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("intervention", intervention_node)

    # Only add ToolNode if tools are provided
    if tools:
        workflow.add_node("tools", ToolNode(tools))
    else:
        # Dummy tool node if none provided to satisfy edge definition (or handle conditionally)
        workflow.add_node("tools", ToolNode([]))

    # Define Edges
    workflow.add_edge(START, "risk_detector")

    workflow.add_conditional_edges(
        "risk_detector",
        intervention_gate,
        {
            "intervention": "intervention",
            "agent": "agent"
        }
    )

    if tools:
        workflow.add_conditional_edges(
            "agent",
            route_tools,
            {
                "tools": "tools",
                "__end__": END
            }
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)

    workflow.add_edge("intervention", END)

    return workflow.compile(checkpointer=MemorySaver())
