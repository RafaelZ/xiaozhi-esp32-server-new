from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage

class UserProfile(TypedDict):
    """User profile containing psychological state summary."""
    user_id: str
    risk_history: List[str]
    daily_summaries: List[str]
    last_summary_time: Optional[float]

class AgentState(TypedDict):
    """State of the agent conversation."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    risk_level: str  # "LOW", "HIGH"
    user_profile: Optional[UserProfile]
    # We might need to carry the connection handler or config if not using config
    # But usually we pass those via 'config' in invoke, or assume they are available in the node context
    # For simplicity, we can store temporary variables here
    current_tool_calls: Optional[List[Dict[str, Any]]]
    llm_output: Optional[str]
