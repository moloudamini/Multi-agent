# LangGraph Multi-Agent API

A production-ready multi-agent workflow system using **LangGraph**, **FastAPI**, **SQLite**, and **Ollama**.

## Features

- **Multi-Agent Orchestration**: Specialized agents (Supervisor, Researcher, Writer) working together.
- **Persistent Storage**: Session persistence powered by **SQLite** (`AsyncSqliteSaver`), making it easy to run locally without external dependencies.
- **Local LLMs**: Integration with **Ollama** for privacy and cost-efficiency.
- **Native Tooling**: Uses LangGraph's `ToolNode` and standard `@tool` decorators.
- **Streaming**: Real-time response streaming via Server-Sent Events (SSE).

## Architecture

1.  **Supervisor**: Uses structured output (Pydantic) to route between specialists.
2.  **Researcher**: Information gathering agent equipped with tools.
3.  **ToolNode**: Executes tool calls (Web Search, Calculator, File Reader).
4.  **Writer**: Synthesizes final reports from gathered data.

## Setup

### Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- Python 3.10+.

### Installation

1.  **Install dependencies**:
    ```bash
    uv pip install -e .
    ```

2.  **Configure Environment**:
    Create a `.env` file in the `p1` directory:
    ```env
    OLLAMA_MODEL=llama3.2
    SQLITE_DB_PATH=agents.db
    ```

### Running the API

```bash
python -m uvicorn app.main:app --reload
```

## API Endpoints

- `POST /api/chat`: Send a message and get a persistent response.
- `POST /api/chat/stream`: Stream response tokens in real-time.
- `GET /api/tools`: List available agent tools.
- `GET /health`: System health check.

## Persistence

This system uses `AsyncSqliteSaver` for checkpointing. To resume a conversation, simply provide the same `session_id` in your requests. The conversation history and agent state are automatically retrieved from the `agents.db` file.
