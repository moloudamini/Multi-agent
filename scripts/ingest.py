from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def ingest_documents(path: str = "knowledge_base"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    # Load documents from the specified directory
    loader = DirectoryLoader(
        path=path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings using Ollama
    embed_model = OllamaEmbeddings(model="llama3.2")

    # Store in ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory="db/chroma",
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Ingested {len(chunks)} document chunks into ChromaDB.")
    return vector_store


if __name__ == "__main__":
    ingest_documents()
