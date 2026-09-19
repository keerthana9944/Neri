import chromadb

from src.config import (
    COLLECTION_NAME,
    VECTOR_DIR,
)


# ============================================================
# ChromaDB Client
# ============================================================

client = chromadb.PersistentClient(
    path=str(VECTOR_DIR)
)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def get_collection():
    """Return the ChromaDB collection instance."""
    return collection



# ============================================================
# Add Documents
# ============================================================

def add_documents(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Add or update document chunks in ChromaDB.
    """

    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    ids = [
        chunk["metadata"]["chunk_id"]
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ============================================================
# General Search
# ============================================================

def search_documents(
    embedding: list[float],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Search ChromaDB using an embedding.

    An optional metadata filter can be supplied.
    """

    query_arguments = {
        "query_embeddings": [embedding],
        "n_results": n_results,
    }

    if where:
        query_arguments["where"] = where

    return collection.query(
        **query_arguments
    )


# ============================================================
# Document Count
# ============================================================

def get_document_count() -> int:
    """
    Return the total number of indexed chunks.
    """

    return collection.count()