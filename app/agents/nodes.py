"""Agent node implementations for the LangGraph workflow using structured outputs."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..models import get_llm
from ..tools import get_all_tools
from .state import AgentState

logger = logging.getLogger(__name__)


class Router(BaseModel):
    """Decide which agent to call next based on the current progress."""

    next_action: Literal["researcher", "writer", "end"] = Field(
        description="The next agent to act. 'researcher' for info gathering, 'writer' for final report, or 'end' when done."
    )


async def supervisor_node(state: AgentState) -> dict:
    """Supervisor node that routes to appropriate worker agents using structured output.

    Analyzes the task and decides which agent should handle it next.

    Args:
        state: Current workflow state

    Returns:
        Updated state with next_action determined via LLM structured output
    """
    llm = get_llm().with_structured_output(Router)

    system_prompt = """You are a supervisor managing a team of agents.
Your job is to analyze the task and route to the appropriate agent:
- 'researcher': For information gathering, search, or retrieval
- 'writer': For synthesizing results into a final response
- 'end': If the user's request has been fully addressed
"""

    messages = state["messages"]
    current_task = state.get("current_task", "")

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Current task: {current_task}\nRecent messages: {messages[-2:] if len(messages) > 1 else messages}"
                ),
            ]
        )

        action = response.next_action
        logger.info(f"Supervisor routed to: {action}")
        return {"next_action": action}

    except Exception as e:
        logger.error(f"Supervisor error: {e}")
        # Default to researcher on error
        return {"next_action": "researcher"}


async def researcher_node(state: AgentState) -> dict:
    """Researcher agent that gathers information using tools.

    Args:
        state: Current workflow state

    Returns:
        Updated state with research results
    """
    tools = get_all_tools()
    llm = get_llm().bind_tools(tools)

    system_prompt = """You are a researcher. Use tools to find information.

Tool selection guide:
- Use `rag_search` FIRST for questions about specific companies, people, or topics (e.g. Google, Apple, investors). It searches the internal knowledge base.
- Use `web_search` for current events, general knowledge, or when rag_search returns no useful results.
- Use `calculator` for math.
- Use `file_reader` to read local files.

Always try `rag_search` before `web_search` when the query is about a named entity or topic."""

    messages = [SystemMessage(content=system_prompt), *state["messages"]]

    try:
        response = await llm.ainvoke(messages)
        return {
            "messages": [response],
            "current_task": "Researching info",
            "tool_results": state.get("tool_results", {}),
            "agent": "researcher",
        }
    except Exception as e:
        logger.error(f"Researcher error: {e}")
        return {"messages": [AIMessage(content=f"Research error: {e}")]}


async def writer_node(state: AgentState) -> dict:
    """Writer agent that synthesizes results into final output.

    Args:
        state: Current workflow state

    Returns:
        Updated state with written output
    """
    llm = get_llm()

    system_prompt = """You are a writer. Synthesize the research results into a clear, well-structured answer.

Rules:
- Use ONLY information from the research messages above. Do not add outside knowledge.
- If the research did not find an answer, say so clearly instead of guessing.
- Be concise and factual. Do not speculate or extrapolate beyond what was found."""

    messages = [SystemMessage(content=system_prompt), *state["messages"]]

    try:
        response = await llm.ainvoke(messages)
        logger.info("Writer completed synthesis")
        return {
            "messages": [response],
            "current_task": "Writing completed",
            "agent": "writer",
        }
    except Exception as e:
        logger.error(f"Writer error: {e}")
        return {"messages": [AIMessage(content=f"Writing error: {e}")]}
