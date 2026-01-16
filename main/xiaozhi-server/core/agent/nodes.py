import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from core.agent.state import AgentState

# Placeholder for the actual LLM initialization logic
# In a real implementation, this would come from the config/provider
def get_llm():
    # Example: return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return ChatOpenAI(model="gpt-4o", temperature=0.7)

class RiskGuardNode:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a psychological risk assessment expert. "
                       "Analyze the user's input for signs of suicide, self-harm, or severe depression. "
                       "Return valid JSON with keys: 'risk_level' (safe, low, high) and 'reason'."),
            ("user", "{input}")
        ])

    async def __call__(self, state: AgentState):
        messages = state["messages"]
        last_user_message = messages[-1].content if messages else ""

        # Simple check or LLM call
        chain = self.prompt | self.llm
        try:
            # Force JSON mode in real implementation
            response = await chain.ainvoke({"input": last_user_message})
            content = response.content.strip()
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:-3]

            risk_data = json.loads(content)
            return {
                "risk_level": risk_data.get("risk_level", "safe"),
                "risk_reason": risk_data.get("reason", "")
            }
        except Exception as e:
            # Fallback to safe
            return {"risk_level": "safe", "risk_reason": "Error in risk check"}

class InterventionNode:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    async def __call__(self, state: AgentState):
        risk_reason = state.get("risk_reason", "Detected risk")

        system_msg = (f"The user is exhibiting high psychological risk ({risk_reason}). "
                      "Provide an empathetic, supportive response. "
                      "Encourage them to seek professional help if necessary. "
                      "Do NOT use tools. Be gentle.")

        messages = [SystemMessage(content=system_msg)] + state["messages"]
        response = await self.llm.ainvoke(messages)

        return {"messages": [response]}

class AgentNode:
    def __init__(self, llm=None, tools=None):
        self.llm = llm or get_llm()
        self.tools = tools or []
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    async def __call__(self, state: AgentState):
        messages = state["messages"]
        # In a real app, you'd prepend the System Prompt from config here
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}
