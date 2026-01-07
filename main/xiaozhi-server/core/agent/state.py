from typing import Annotated, List, Optional, Literal, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    """
    The state of the agent in the graph.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_id: str
    risk_level: Optional[str]  # 'low', 'medium', 'high'
    is_interrupted: bool
    user_profile: Optional[Dict[str, Any]]  # For long-term memory
