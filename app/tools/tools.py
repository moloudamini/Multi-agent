"""Example custom tools for the multi-agent system using LangChain native @tool."""

from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool


@tool
def web_search(query: Annotated[str, "The search query to look up information for."]):
    """Search the web for information about a topic."""
    tavily = TavilySearch(max_results=3)
    results = tavily.invoke(query)
    return f"Search results for '{query}':\n{results}"


@tool
def calculator(
    expression: Annotated[
        str, "A mathematical expression to evaluate (e.g., '2 + 2' or '15 * 4')."
    ],
):
    """Perform mathematical calculations. Use this for any math or logic tasks."""
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/().% ")
        if all(c in allowed_chars for c in expression):
            # Note: In production, use a library like 'numexpr' or a safer sandbox
            result = eval(expression)
            return f"Result: {result}"
        return "Error: Invalid characters in expression"
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def file_reader(file_path: Annotated[str, "The path to the file to be read."]):
    """Read contents of a file. Use this when you need to access local data or configuration files."""
    try:
        with open(file_path) as f:
            content = f.read(1000)  # Read first 1000 chars
        return f"Contents of {file_path}:\n{content}"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


# Registry of all tools
_all_tools = [web_search, calculator, file_reader]


def get_all_tools():
    """Get all available tools.

    Returns:
        List of LangChain tool instances
    """
    return _all_tools


def get_tools_by_name():
    """Get tools indexed by name for efficient lookup.

    Returns:
        Dictionary mapping tool names to tool instances
    """
    return {tool.name: tool for tool in _all_tools}
