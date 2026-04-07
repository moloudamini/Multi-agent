"""FastAPI application entry point with SQLite persistence."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .models import get_llm
from .routers import agent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # 1. Setup LangGraph SQLite Checkpointer
    async with AsyncSqliteSaver.from_conn_string("agents.sqlite") as checkpointer:
        app.state.checkpointer = checkpointer

        # 2. Verify Ollama connection

        try:
            llm = get_llm()
            await llm.ainvoke("ping")
            logger.info("Successfully verified connection to Ollama.")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")

        yield

    logger.info("SQLite checkpointer closed.")


app = FastAPI(
    title="LangGraph Multi-Agent API",
    description="Multi-agent workflow system using LangGraph, SQLite, and Ollama",
    version="0.3.1",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(agent.router, prefix="/api", tags=["agent"])
