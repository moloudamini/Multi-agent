"""LangGraph workflow definition for multi-agent system using native abstractions and persistent storage."""

import logging

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ..tools import get_all_tools
from .nodes import researcher_node, supervisor_node, writer_node
from .state import AgentState

logger = logging.getLogger(__name__)


def create_agent_graph(checkpointer=None):
    """Create and compile the multi-agent LangGraph workflow.

    Args:
        checkpointer: The checkpointer for state persistence (e.g., AsyncPostgresSaver).
    """
    workflow = StateGraph(AgentState)

    # 1. Add nodes
    tools = get_all_tools()
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("tool_executor", ToolNode(tools))

    # 2. Set entry point
    workflow.set_entry_point("supervisor")

    # 3. Add edges from supervisor (Routing Decision)
    def route_supervisor(state: AgentState) -> str:
        """Route based on structured supervisor decision."""
        # Return the key that corresponds to the mapping below
        return state.get("next_action", "end")

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "writer": "writer", "end": END},
    )

    # 4. Add edges from researcher
    def route_researcher(state: AgentState) -> str:
        """Route based on if the researcher called tools."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_executor"
        return "supervisor"

    workflow.add_conditional_edges(
        "researcher",
        route_researcher,
        {"tool_executor": "tool_executor", "supervisor": "supervisor"},
    )

    # 5. Tool executor always returns to the researcher
    workflow.add_edge("tool_executor", "researcher")

    # 6. Writer completes the workflow
    workflow.add_edge("writer", END)

    # Compile with persistence
    return workflow.compile(checkpointer=checkpointer)
