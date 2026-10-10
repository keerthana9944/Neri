from src.embeddings import create_embedding
from src.machine_norm import (
    get_machine_retrieval_aliases,
    normalize_machine_id,
)
from src.vector_store import search_documents


# ============================================================
# Helper
# ============================================================

def _convert_results(
    results: dict,
) -> list[dict]:
    """
    Convert ChromaDB results into Neri's
    internal retrieval format.
    """

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved_chunks.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_chunks


# ============================================================
# Main Retrieval
# ============================================================

def retrieve_documents(
    query: str,
    machine: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Retrieve approved documentation for a troubleshooting query.

    Machine Pre-Filtering:
        Pre-filters ChromaDB retrieval using the supported 'where' mechanism
        BEFORE semantic ranking, ensuring machine isolation and preventing
        irrelevant documents from other machines from crowding out results.
        Supports canonical machine IDs and backward-compatible legacy aliases.
        Never falls back to cross-machine evidence when a specific machine is requested.

    Safety-Critical Retrieval:
        In addition to normal retrieval, explicitly searches the selected
        machine's approved safety documentation using ChromaDB metadata pre-filtering.
    """

    if not query.strip():
        return []

    query_embedding = create_embedding(
        query
    )

    # Determine canonical machine and retrieval variants for ChromaDB pre-filtering
    has_specific_machine = bool(machine and str(machine).strip() and str(machine).strip().lower() != "unknown")
    machine_variants = get_machine_retrieval_aliases(machine) if has_specific_machine else []

    # ========================================================
    # Normal Machine Retrieval (Pre-filtered in ChromaDB)
    # ========================================================

    normal_where = None
    if has_specific_machine and machine_variants:
        normal_where = {
            "machine": {
                "$in": machine_variants
            }
        }

    results = search_documents(
        embedding=query_embedding,
        n_results=n_results,
        where=normal_where,
    )

    normal_chunks = _convert_results(
        results
    )

    # ========================================================
    # Explicit Safety Retrieval (Pre-filtered in ChromaDB)
    # ========================================================

    safety_chunks = []

    if has_specific_machine and machine_variants:
        safety_where = {
            "$and": [
                {
                    "machine": {
                        "$in": machine_variants
                    }
                },
                {
                    "document_type": {
                        "$eq": "safety"
                    }
                },
            ]
        }

        safety_results = search_documents(
            embedding=query_embedding,
            n_results=3,
            where=safety_where,
        )

        safety_chunks = _convert_results(
            safety_results
        )
    elif not has_specific_machine:
        # If no specific machine is required, query general safety documentation
        general_safety_where = {
            "document_type": {
                "$eq": "safety"
            }
        }

        safety_results = search_documents(
            embedding=query_embedding,
            n_results=3,
            where=general_safety_where,
        )

        safety_chunks = _convert_results(
            safety_results
        )

    # ========================================================
    # Combine Results
    # ========================================================

    combined_chunks = []
    seen_chunk_ids = set()

    # Add normal results first
    for chunk in normal_chunks:
        chunk_id = chunk.get(
            "metadata",
            {},
        ).get(
            "chunk_id"
        )

        ident = chunk_id or str(hash(chunk.get("text", "")))

        if ident not in seen_chunk_ids:
            combined_chunks.append(chunk)
            seen_chunk_ids.add(ident)

    # Add explicit safety results
    for chunk in safety_chunks:
        chunk_id = chunk.get(
            "metadata",
            {},
        ).get(
            "chunk_id"
        )

        ident = chunk_id or str(hash(chunk.get("text", "")))

        if ident not in seen_chunk_ids:
            combined_chunks.append(chunk)
            seen_chunk_ids.add(ident)

    return combined_chunks