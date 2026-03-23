"""LLM configuration."""

from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    """Get a ChatOllama instance."""
    return ChatOllama(
        model="llama3.2",
        temperature=0.7,
    )
