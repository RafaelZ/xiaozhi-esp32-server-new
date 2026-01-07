from typing import List, Dict, Any, Callable
from langchain_core.tools import Tool
from core.providers.tools.unified_tool_handler import UnifiedToolHandler

class ToolAdapter:
    """
    Adapts the existing UnifiedToolHandler to LangChain Tools.
    """
    def __init__(self, unified_handler: UnifiedToolHandler):
        self.unified_handler = unified_handler

    def get_langchain_tools(self) -> List[Tool]:
        """
        Converts registered tools in UnifiedToolHandler to LangChain Tool objects.
        """
        tools = []
        # Get descriptions from the handler
        # The existing get_functions returns a list of Dicts matching OpenAI function schema
        descriptions = self.unified_handler.get_functions()

        for desc in descriptions:
            name = desc["name"]
            description = desc.get("description", "")

            # We need to create a callable that wraps the execute_tool method
            async def _wrapper(**kwargs):
                # The unified handler expects 'arguments' as a dict
                # LangChain passes kwargs directly
                try:
                    result = await self.unified_handler.tool_manager.execute_tool(name, kwargs)
                    if result.response:
                        return result.response
                    return result.result or "Tool executed successfully with no output."
                except Exception as e:
                    return f"Error executing tool {name}: {str(e)}"

            tool = Tool(
                name=name,
                description=description,
                func=None, # We use coroutine for async
                coroutine=_wrapper
            )
            tools.append(tool)

        return tools
