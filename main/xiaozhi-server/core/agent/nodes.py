import json
import uuid
import asyncio
from typing import Literal, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from core.agent.state import AgentState
# from core.utils.llm import LLM  # The existing LLM abstract class
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from core.handle.textHandle import handleTextMessage
from core.utils import textUtils
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from plugins_func.register import Action

# System prompt for risk detection
RISK_SYSTEM_PROMPT = """
You are a psychological risk assessment expert.
Analyze the user's latest input for signs of self-harm, suicide, severe depression, or imminent danger.
Output ONLY one word: "HIGH" or "LOW".
"""

# System prompt for intervention
INTERVENTION_SYSTEM_PROMPT = """
You are a compassionate crisis intervention specialist.
The user has been flagged as high risk (self-harm, suicide, etc.).
Your goal is to:
1. Empathize with the user's pain.
2. De-escalate the situation.
3. Gently encourage seeking professional help.
Do not be judgmental. Keep your response short, calm, and supportive.
"""

class AgentNodes:
    def __init__(self, conn):
        """
        Initialize nodes with the connection handler.
        We need 'conn' to access the existing LLM, TTS, Memory, and ToolHandler.
        """
        self.conn = conn
        self.llm = conn.llm
        self.tool_handler = conn.func_handler
        self.logger = conn.logger
        self.loop = conn.loop or asyncio.get_event_loop()

    async def _run_llm_in_executor(self, method, *args, **kwargs):
        """Helper to run blocking LLM calls in executor"""
        return await self.loop.run_in_executor(None, lambda: method(*args, **kwargs))

    async def risk_guard_node(self, state: AgentState, config: RunnableConfig):
        """
        Analyzes the user input for psychological risks.
        """
        user_input = state["user_input"]
        messages = [{"role": "system", "content": RISK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}]

        session_id = f"{self.conn.session_id}_risk_check"

        try:
            # Run blocking call in executor
            # Note: self.llm.response returns a generator/iterator.
            # We need to consume it fully inside the executor or handle streaming.
            # For risk check, we don't need streaming.

            def get_response():
                response_iter = self.llm.response(session_id, messages)
                return "".join([chunk for chunk in response_iter])

            risk_assessment = await self.loop.run_in_executor(None, get_response)

            risk_assessment = risk_assessment.strip().upper()
            if "HIGH" in risk_assessment:
                return {"risk_level": "HIGH"}
            else:
                return {"risk_level": "LOW"}
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {"risk_level": "LOW"}

    async def psych_intervention_node(self, state: AgentState, config: RunnableConfig):
        """
        Generates an intervention response for high-risk users.
        """
        user_input = state["user_input"]
        messages = [{"role": "system", "content": INTERVENTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input}]

        session_id = f"{self.conn.session_id}_intervention"

        self.conn.sentence_id = str(uuid.uuid4().hex)
        full_response = ""

        # We can't easily stream from inside run_in_executor if the underlying generator is blocking.
        # But we can iterate.
        # Since we are refactoring, we assume we can block for the whole response for intervention,
        # or we accept blocking the loop slightly if we iterate directly.
        # Ideally, LLM client should be async.

        def get_response():
            return list(self.llm.response(session_id, messages))

        chunks = await self.loop.run_in_executor(None, get_response)

        for chunk in chunks:
            full_response += chunk
            # Stream to TTS queue (thread-safe queue)
            self.conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=self.conn.sentence_id,
                    sentence_type=SentenceType.MIDDLE,
                    content_type=ContentType.TEXT,
                    content_detail=chunk,
                )
            )

        return {"messages": [AIMessage(content=full_response)], "llm_output": full_response}

    async def main_agent_node(self, state: AgentState, config: RunnableConfig):
        """
        The main conversational agent.
        """
        messages = state["messages"]

        formatted_messages = []
        # Add system prompt
        if self.conn.prompt:
             formatted_messages.append({"role": "system", "content": self.conn.prompt})

        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                content = msg.content
                if msg.tool_calls:
                     # Reconstruct tool calls for LLM context if needed
                     # This depends on what self.llm expects.
                     # Usually openai-compatible expects 'tool_calls' field.
                     # Here we simplify.
                     pass
                formatted_messages.append({"role": "assistant", "content": content})
            elif isinstance(msg, ToolMessage):
                formatted_messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})

        functions = self.tool_handler.get_functions()
        session_id = self.conn.session_id

        self.conn.sentence_id = str(uuid.uuid4().hex)

        try:
            # We need to handle the generator which yields (content, tool_calls) tuples
            # Blocking call
            def get_response_items():
                return list(self.llm.response_with_functions(session_id, formatted_messages, functions=functions))

            response_items = await self.loop.run_in_executor(None, get_response_items)

            full_content = ""
            tool_calls_list = [] # List of dicts

            # Helper to merge tool calls similar to ConnectionHandler
            def merge_tool_calls(target_list, new_calls):
                 for tool_call in new_calls:
                    # tool_call is an object/Namespace, usually from OpenAI library
                    # We need to access attributes carefully

                    # Check if object or dict
                    t_index = getattr(tool_call, "index", None)
                    t_id = getattr(tool_call, "id", None)
                    t_func = getattr(tool_call, "function", None)
                    t_name = getattr(t_func, "name", None) if t_func else None
                    t_args = getattr(t_func, "arguments", None) if t_func else None

                    if t_index is None:
                        t_index = len(target_list) if t_name else len(target_list) - 1

                    if t_index >= len(target_list):
                        target_list.append({"id": "", "name": "", "arguments": ""})

                    if t_id: target_list[t_index]["id"] = t_id
                    if t_name: target_list[t_index]["name"] = t_name
                    if t_args: target_list[t_index]["arguments"] += t_args

            for item in response_items:
                content = None
                tools = None

                if isinstance(item, tuple):
                    content, tools = item
                elif isinstance(item, dict):
                    content = item.get("content")
                    tools = item.get("tool_calls")
                else:
                    content = item

                if content:
                    full_content += content
                    self.conn.tts.tts_text_queue.put(
                        TTSMessageDTO(
                            sentence_id=self.conn.sentence_id,
                            sentence_type=SentenceType.MIDDLE,
                            content_type=ContentType.TEXT,
                            content_detail=content,
                        )
                    )

                if tools:
                    if isinstance(tools, list):
                        merge_tool_calls(tool_calls_list, tools)

            if tool_calls_list:
                 lc_tool_calls = []
                 for tc in tool_calls_list:
                     try:
                         args = json.loads(tc["arguments"])
                     except:
                         args = {}

                     lc_tool_calls.append({
                         "id": tc["id"] or str(uuid.uuid4()),
                         "name": tc["name"],
                         "args": args,
                         "type": "tool_call"
                     })

                 return {"messages": [AIMessage(content=full_content, tool_calls=lc_tool_calls)]}

            return {"messages": [AIMessage(content=full_content)]}

        except Exception as e:
            self.logger.error(f"Main agent failed: {e}")
            return {"messages": [AIMessage(content="I'm sorry, I encountered an error.")]}

    async def tool_node(self, state: AgentState, config: RunnableConfig):
        """
        Executes tools requested by the MainAgent.
        """
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        tool_results = []
        for tool_call in last_message.tool_calls:
            function_name = tool_call["name"]
            arguments = tool_call["args"]
            tool_call_id = tool_call["id"]

            tool_data = {
                "name": function_name,
                "arguments": arguments,
                "id": tool_call_id
            }

            # handle_llm_function_call is async
            result_obj = await self.tool_handler.handle_llm_function_call(self.conn, tool_data)

            content = result_obj.result if result_obj.result else str(result_obj.response)

            tool_results.append(ToolMessage(tool_call_id=tool_call_id, content=content, name=function_name))

        return {"messages": tool_results}
