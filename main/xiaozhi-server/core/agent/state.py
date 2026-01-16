from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    The state of the agent in the LangGraph.
    """
    # The list of messages in the conversation
    messages: Annotated[List[BaseMessage], operator.add]

    # User's profile and psychological state
    user_profile: Dict[str, Any]

    # Current risk assessment
    risk_level: str  # "safe", "low", "high"
    risk_reason: str

    # Session ID for logging/persistence
    session_id: str
