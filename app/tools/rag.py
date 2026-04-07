"""RAG tool using ChromaDB and Ollama embeddings."""

import logging
from typing import Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _get_retriever():
    """Get retriever, raising clearly if unavailable."""
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as e:
        raise RuntimeError("chromadb is not installed. Run: uv add chromadb") from e

    embed_model = OllamaEmbeddings(
        model="llama3.2",
    )
    vector_store = Chroma(
        persist_directory="db/chroma",
        embedding_function=embed_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 10, "lambda_mlt": 0.5}
    )

    return retriever


@tool
def rag_search(
    query: Annotated[str, "The question or topic to search the knowledge base for."],
):
    """Search the local knowledge base for relevant information using semantic similarity.
    Use this when the user asks about topics that may be in internal documents."""
    try:
        retriever = _get_retriever()
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."

        content = (
            f"Answer this question: {query} only based on the following information\n\n"
            + "\n\n".join(
                [f"{i + 1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
            )
        )
        llm = ChatOllama(model="llama3.2")
        results = llm.invoke([HumanMessage(content=content)])
        logger.info(
            f"RAG search successful for query: {query} based on this content: {content}"
        )
        return results.content

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Error searching knowledge base: {e}"
