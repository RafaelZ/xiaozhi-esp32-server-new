from langgraph.graph import StateGraph, END
from core.agent.state import AgentState
from core.agent.nodes import AgentNodes

def create_agent_graph(conn):
    """
    Creates and compiles the LangGraph application.
    """
    nodes = AgentNodes(conn)
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("risk_guard", nodes.risk_guard_node)
    workflow.add_node("main_agent", nodes.main_agent_node)
    workflow.add_node("psych_intervention", nodes.psych_intervention_node)
    workflow.add_node("tool_executor", nodes.tool_node)

    # Define Conditional Logic
    def route_risk(state: AgentState):
        if state["risk_level"] == "HIGH":
            return "psych_intervention"
        return "main_agent"

    def route_tools(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_executor"
        return END

    # Add Edges
    workflow.set_entry_point("risk_guard")

    workflow.add_conditional_edges(
        "risk_guard",
        route_risk,
        {
            "psych_intervention": "psych_intervention",
            "main_agent": "main_agent"
        }
    )

    workflow.add_conditional_edges(
        "main_agent",
        route_tools,
        {
            "tool_executor": "tool_executor",
            END: END
        }
    )

    workflow.add_edge("tool_executor", "main_agent")
    workflow.add_edge("psych_intervention", END)

    # Compile
    app = workflow.compile()
    return app
