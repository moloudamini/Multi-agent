"""FastAPI routes for persistent agent interaction."""

import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from ..agents import AgentState
from ..agents.graph import create_agent_graph

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = Field(default=None)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    agent: str
    response: str
    tool_results: dict[str, str]
    session_id: str


class ToolInfo(BaseModel):
    """Information about an available tool."""

    name: str
    description: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Send a message to the agent workflow with PostgreSQL persistence."""
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Processing chat for session {session_id}")

    try:
        # 1. Get checkpointer from app state
        checkpointer = req.app.state.checkpointer

        # 2. Create the graph with persistence
        graph = create_agent_graph(checkpointer=checkpointer)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=request.message)],
            "next_action": "researcher",
            "tool_results": {},
            "current_task": request.message,
        }

        # 3. Configure the thread (PostgreSQL uses thread_id for isolation)
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        # 4. Invoke the graph
        result = await graph.ainvoke(initial_state, config=config)

        # 5. Extract final response and trajectory
        last_message = result["messages"][-1]
        response_content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        return ChatResponse(
            agent=result.get("agent", "unknown"),
            response=response_content,
            tool_results=result.get("tool_results", {}),
            session_id=session_id,
        )
    except Exception as e:
        logger.exception(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """Stream agent response tokens with PostgreSQL persistence."""
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Starting stream for session {session_id}")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            checkpointer = req.app.state.checkpointer
            graph = create_agent_graph(checkpointer=checkpointer)

            initial_state: AgentState = {
                "messages": [HumanMessage(content=request.message)],
                "next_action": "researcher",
                "tool_results": {},
                "current_task": request.message,
            }

            config: RunnableConfig = {"configurable": {"thread_id": session_id}}

            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                if await req.is_disconnected():
                    logger.info(f"Client disconnected: {session_id}")
                    break

                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        yield f"data: {chunk.content}\n\n"
                if kind == "on_chain_start":
                    input = event["data"]["input"]
                    if input and isinstance(input, dict):
                        messages = input.get("messages", [])
                        if messages:
                            last_message = messages[-1]
                            if isinstance(last_message, ToolMessage):
                                yield f"data: [TOOL] {last_message.content}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(), media_type="text/event-stream", headers={"X-Session-ID": session_id}
    )


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """List all available tools."""
    from ..tools import get_all_tools

    return [
        ToolInfo(name=tool.name, description=tool.description)
        for tool in get_all_tools()
    ]


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}
