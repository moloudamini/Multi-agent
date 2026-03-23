import asyncio

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agents.graph import create_agent_graph
from app.tools.tools import calculator, web_search

load_dotenv()


@pytest.mark.asyncio
async def test_web_search_tool():
    """Test if Tavily search tool is working."""
    try:
        result = await web_search.ainvoke({"query": "current weather in Tokyo"})
        assert "Search results" in result
        print("✅ Web Search Tool: Working")
    except Exception as e:
        pytest.fail(f"Web Search Tool failed: {e}")


@pytest.mark.asyncio
async def test_calculator_tool():
    """Test if the calculator tool is working."""
    result = await calculator.ainvoke({"expression": "15 * 4 + 10"})
    assert "70" in result
    print("✅ Calculator Tool: Working")


@pytest.mark.asyncio
async def test_full_graph_workflow():
    """Test the full supervisor -> researcher -> writer flow."""
    graph = create_agent_graph(checkpointer=None)
    initial_state = {
        "messages": [HumanMessage(content="Who is the current CEO of Nvidia?")],
        "next_action": "supervisor",
        "tool_results": {},
        "current_task": "Identify CEO",
    }
    try:
        result = await graph.ainvoke(initial_state)
        assert len(result["messages"]) > 1
        print(
            "✅ Basic Graph Workflow: Working with response:",
            result["messages"][-1].content,
        )
    except Exception as e:
        pytest.fail(f"Graph Workflow failed: {e}")


@pytest.mark.asyncio
async def test_multi_step_research_workflow():
    """Test a scenario requiring multiple tool calls before synthesis."""
    graph = create_agent_graph(checkpointer=None)
    query = "Compare the market cap of Apple and Microsoft as of their last earnings reports."
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_action": "supervisor",
        "tool_results": {},
        "current_task": "Compare market caps",
    }

    print(f"\nTesting Multi-step Research: '{query}'")
    result = await graph.ainvoke(initial_state)
    has_tool_calls = any(
        hasattr(m, "tool_calls") and m.tool_calls for m in result["messages"]
    )
    assert has_tool_calls, "Researcher should have used tools for this query"
    print(
        f"✅ Multi-step Research: Working with response: {result['messages'][-1].content}"
    )


@pytest.mark.asyncio
async def test_direct_writer_synthesis():
    """Test if supervisor can skip research if info is already present."""
    graph = create_agent_graph(checkpointer=None)
    initial_state = {
        "messages": [
            HumanMessage(
                content="Write a poem about the following data: The sky is blue, grass is green."
            ),
        ],
        "next_action": "supervisor",
        "tool_results": {},
        "current_task": "Write a poem",
    }

    print("\nTesting Direct Synthesis (Skip Research)...")
    result = await graph.ainvoke(initial_state)
    assert len(result["messages"]) >= 2
    print("✅ Direct Synthesis: Working with response:", result["messages"][-1].content)


@pytest.mark.asyncio
async def test_math_and_research_combo():
    """Test using both the calculator and web search in one flow."""
    graph = create_agent_graph(checkpointer=None)
    query = "Find the price of Bitcoin and tell me what 0.5 BTC is worth."
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_action": "supervisor",
        "tool_results": {},
        "current_task": "Price lookup and calculation",
    }

    print(f"\nTesting Combo Flow: '{query}'")
    result = await graph.ainvoke(initial_state)
    final_content = result["messages"][-1].content
    print(f"Final Content: {final_content}")
    assert any(char.isdigit() for char in final_content)
    print("✅ Combo Flow: Working with response:", final_content)


if __name__ == "__main__":

    async def run_tests():
        print("Starting Extended Agent Tests...\n")
        try:
            await test_web_search_tool()
            await test_calculator_tool()
            await test_full_graph_workflow()
            await test_multi_step_research_workflow()
            await test_direct_writer_synthesis()
            await test_math_and_research_combo()
            print("\nAll tests completed successfully.")
        except Exception as e:
            print(f"\nTests failed with error: {e}")

    asyncio.run(run_tests())
