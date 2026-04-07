# LangGraph Multi-Agent API

A production-ready multi-agent workflow system using **LangGraph**, **FastAPI**, **SQLite**, and **Ollama**.

## Features

- **Multi-Agent Orchestration**: Specialized agents (Supervisor, Researcher, Writer) working together.
- **Persistent Storage**: Session persistence powered by **SQLite** (`AsyncSqliteSaver`), making it easy to run locally without external dependencies.
- **Local LLMs**: Integration with **Ollama** for privacy and cost-efficiency.
- **Native Tooling**: Uses LangGraph's `ToolNode` and standard `@tool` decorators.
- **Streaming**: Real-time response streaming via Server-Sent Events (SSE).
- **RAG Search**: Semantic search over local documents via ChromaDB and Ollama embeddings.

## Architecture

```
User Request → Supervisor → Researcher (with tools) → Supervisor → Writer → END
                         ↗ (tool calls)
                    ToolNode
```

- **Supervisor**: Uses structured output (Pydantic) to route between specialists (`researcher`, `writer`, or `end`).
- **Researcher**: Information-gathering agent equipped with tools; loops back to supervisor after gathering info.
- **ToolNode**: LangGraph prebuilt node that executes tool calls and returns results to the researcher.
- **Writer**: Synthesizes a final report from gathered data; always terminates the graph.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running with the `llama3.2` model:
  ```bash
  ollama pull llama3.2
  ```
- A `TAVILY_API_KEY` for web search (get one at [tavily.com](https://tavily.com)).

### Installation

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Install with dev dependencies**:
   ```bash
   uv sync --extra dev
   ```

3. **Configure environment** — create a `.env` file in the project root:
   ```env
   TAVILY_API_KEY=your_key_here
   ```

### Ingest Knowledge Base (optional)

Place documents (`.txt`, `.pdf`, etc.) in `knowledge_base/` then run the ingestion script to populate the Chroma vector store:

```bash
uv run python scripts/ingest.py
```

### Running the API

```bash
uv run uvicorn app.main:app --reload
```

## API Endpoints

<!-- AUTO-GENERATED -->
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Send a message; returns a blocking response. Pass `session_id` for conversation continuity. |
| `POST` | `/api/chat/stream` | Stream response tokens in real-time via SSE (`data: <token>` chunks, ends with `data: [DONE]`). |
| `GET` | `/api/tools` | List available agent tools. |
| `GET` | `/api/health` | Health check. |
<!-- AUTO-GENERATED -->

## Available Tools

<!-- AUTO-GENERATED -->
| Tool | Description |
|------|-------------|
| `web_search` | Search the web via Tavily (requires `TAVILY_API_KEY`). |
| `calculator` | Evaluate mathematical expressions safely using a character whitelist. |
| `file_reader` | Read the first 1000 characters of a local file. |
| `rag_search` | Semantic search over the local knowledge base (ChromaDB + Ollama embeddings). |
<!-- AUTO-GENERATED -->

## Development Commands

<!-- AUTO-GENERATED -->
| Command | Description |
|---------|-------------|
| `uv sync` | Install runtime dependencies |
| `uv sync --extra dev` | Install with dev dependencies (pytest, ruff, httpx) |
| `uv run uvicorn app.main:app --reload` | Run the API server with hot reload |
| `uv run pytest tests/` | Run all tests |
| `uv run pytest tests/test_agents.py::test_calculator_tool -v` | Run a single test |
| `uv run ruff check .` | Lint the codebase |
| `uv run ruff format .` | Format the codebase |
<!-- AUTO-GENERATED -->

## Persistence

Session continuity is handled via SQLite (`agents.sqlite`). Pass the same `session_id` in subsequent requests to resume a conversation — the agent state and message history are automatically restored from the checkpoint store.
