"""State schema for the LangGraph multi-agent workflow."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State schema for the multi-agent workflow.

    Attributes:
        messages: Conversation history with message accumulation.
        next_action: Which agent should handle the next step.
        agent_trajectory: List of agents that have acted in this turn.
        tool_results: Dictionary mapping tool call IDs to their results.
        current_task: Description of the current task being processed.
    """

    messages: Annotated[list, add_messages]
    next_action: Literal["researcher", "writer", "tool_executor", "end"]
    agent: str
    tool_results: dict[str, str]
    current_task: str
