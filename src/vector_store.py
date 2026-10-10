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
    clean_existing: bool = True,
) -> None:
    """
    Add or update document chunks in ChromaDB.
    When clean_existing is True, purges any existing chunks for the same
    doc_id first to prevent duplicate or orphaned chunks.
    """

    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    # Purge any existing chunks for these doc_ids to ensure clean updates
    if clean_existing:
        seen_doc_ids = set()
        for chunk in chunks:
            did = chunk.get("metadata", {}).get("doc_id")
            if did and did not in seen_doc_ids:
                seen_doc_ids.add(did)
                try:
                    collection.delete(where={"doc_id": did})
                except Exception:
                    pass

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
# Delete Document Chunks
# ============================================================

def delete_document_chunks(
    doc_id: str | None = None,
    document_name: str | None = None,
    machine: str | None = None,
) -> int:
    """
    Delete document chunks from ChromaDB.

    When 'doc_id' is provided, deletion is strictly restricted to that unique ID
    and will NEVER delete chunks belonging to other documents or machines.
    Filename-based fallback is used ONLY when doc_id is None, and requires
    'machine' to prevent cross-machine chunk deletion.
    """
    deleted_count = 0

    if doc_id:
        try:
            res = collection.delete(where={"doc_id": doc_id})
            if isinstance(res, dict) and "deleted" in res:
                deleted_count += res.get("deleted", 0)
        except Exception:
            pass
        return deleted_count

    elif document_name and machine:
        try:
            from src.machine_norm import get_machine_retrieval_aliases
            aliases = get_machine_retrieval_aliases(machine)
            if aliases:
                res = collection.delete(
                    where={
                        "$and": [
                            {"document": {"$eq": document_name}},
                            {"machine": {"$in": aliases}},
                        ]
                    }
                )
            else:
                res = collection.delete(
                    where={
                        "$and": [
                            {"document": {"$eq": document_name}},
                            {"machine": {"$eq": machine}},
                        ]
                    }
                )

            if isinstance(res, dict) and "deleted" in res:
                deleted_count += res.get("deleted", 0)
        except Exception:
            pass

    return deleted_count


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
