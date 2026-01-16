from langgraph.graph import StateGraph, END
from core.agent.state import AgentState
from core.agent.nodes import RiskGuardNode, InterventionNode, AgentNode
from langgraph.prebuilt import ToolNode

def create_agent_graph(llm=None, tools=None):
    """
    Constructs the LangGraph for the agent.
    """

    # Initialize Nodes
    risk_guard = RiskGuardNode(llm)
    intervention = InterventionNode(llm)
    agent = AgentNode(llm, tools)
    tool_node = ToolNode(tools) if tools else None

    # Define Graph
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("risk_guard", risk_guard)
    workflow.add_node("intervention", intervention)
    workflow.add_node("agent", agent)
    if tool_node:
        workflow.add_node("tools", tool_node)

    # Define Entry Point
    workflow.set_entry_point("risk_guard")

    # Define Edges
    def check_risk(state: AgentState):
        if state.get("risk_level") == "high":
            return "intervention"
        return "agent"

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges(
        "risk_guard",
        check_risk,
        {
            "intervention": "intervention",
            "agent": "agent"
        }
    )

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    if tool_node:
        workflow.add_edge("tools", "agent")

    workflow.add_edge("intervention", END)

    # Compile
    app = workflow.compile()
    return app
