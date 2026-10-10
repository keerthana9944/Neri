import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import (
    get_supervisor_api_key,
    get_technician_api_key,
)
from src.guardrails import (
    has_safety_evidence,
    is_safety_chunk_relevant_to_hazard,
)
from src.history import (
    delete_document,
    get_document_content,
    get_documents,
    initialize_database,
    save_document,
)
from src.ingestion import create_chunks
from src.machine_norm import (
    KNOWN_MACHINES,
    generate_chunk_id,
    generate_document_id,
    get_machine_retrieval_aliases,
    normalize_machine_id,
    resolve_canonical_machine,
)
from src.retrieval import retrieve_documents
from src.vector_store import (
    add_documents,
    delete_document_chunks,
    get_collection,
    search_documents,
)


@pytest.fixture
def supervisor_client():
    return TestClient(app, headers={"X-API-Key": get_supervisor_api_key()})


@pytest.fixture
def technician_client():
    return TestClient(app, headers={"X-API-Key": get_technician_api_key()})


# ============================================================
# TASK 1: Machine ID Normalization Tests
# ============================================================

def test_canonical_machine_normalization():
    """Verify that known aliases resolve to canonical IDs consistently."""
    assert normalize_machine_id("CNC-001") == "cnc-mill"
    assert normalize_machine_id("cnc mill") == "cnc-mill"
    assert normalize_machine_id("Haas VF-2") == "cnc-mill"
    assert normalize_machine_id("cnc_mill") == "cnc-mill"

    assert normalize_machine_id("LAT-001") == "cnc-lathe"
    assert normalize_machine_id("CNC Lathe") == "cnc-lathe"
    assert normalize_machine_id("DMG Mori") == "cnc-lathe"

    assert normalize_machine_id("PRS-001") == "hydraulic-press"
    assert normalize_machine_id("Hydraulic Press") == "hydraulic-press"
    assert normalize_machine_id("Schuler") == "hydraulic-press"

    assert normalize_machine_id("DRL-001") == "industrial-drill"
    assert normalize_machine_id("Industrial Drill") == "industrial-drill"


def test_unknown_and_missing_machine_ids_safety():
    """Verify missing, empty, or unknown machines are handled safely without merging."""
    # Empty / None inputs return 'unknown'
    assert normalize_machine_id(None) == "unknown"
    assert normalize_machine_id("") == "unknown"
    assert normalize_machine_id("   ") == "unknown"

    # Distinct unknown machines get distinct slugs and are never merged
    slug_a = normalize_machine_id("Custom Laser Cutter 1")
    slug_b = normalize_machine_id("Custom Laser Cutter 2")
    assert slug_a == "custom-laser-cutter-1"
    assert slug_b == "custom-laser-cutter-2"
    assert slug_a != slug_b


def test_resolve_canonical_machine():
    """Verify resolve_canonical_machine safely extracts equipment identity from machine or machine_id."""
    # Known alias in machine_id resolves to canonical
    assert resolve_canonical_machine("Milling Dept", "CNC-001") == "cnc-mill"
    # Known alias in machine resolves
    assert resolve_canonical_machine("Hydraulic Press", "PRS-999") == "hydraulic-press"
    # Custom equipment
    assert resolve_canonical_machine("Plasma Cutter", "PC-01") == "plasma-cutter"
    # Missing both
    assert resolve_canonical_machine(None, None) == "unknown"


def test_get_machine_retrieval_aliases_compatibility():
    """Verify retrieval aliases include canonical slug, display name, and legacy casing."""
    aliases_mill = get_machine_retrieval_aliases("CNC-001")
    assert "cnc-mill" in aliases_mill
    assert "CNC Mill" in aliases_mill

    aliases_lathe = get_machine_retrieval_aliases("CNC Lathe")
    assert "cnc-lathe" in aliases_lathe
    assert "CNC Lathe" in aliases_lathe

    assert get_machine_retrieval_aliases(None) == []


# ============================================================
# TASK 2: ChromaDB Retrieval Pre-Filtering Tests
# ============================================================

def test_retrieval_prefiltering_passes_where_clause():
    """Verify retrieve_documents passes pre-filtering 'where' to search_documents before ranking."""
    with patch("src.retrieval.search_documents") as mock_search, \
         patch("src.retrieval.create_embedding", return_value=[0.1] * 3072):

        mock_search.return_value = {
            "documents": [["Test text"]],
            "metadatas": [[{"machine": "cnc-mill", "chunk_id": "c1", "document_type": "manual"}]],
            "distances": [[0.3]],
        }

        # Query for CNC-001
        chunks = retrieve_documents(query="spindle vibration", machine="CNC-001", n_results=5)

        # Ensure search_documents was called twice (normal and explicit safety)
        assert mock_search.call_count == 2

        # 1st call: normal retrieval pre-filter
        normal_where = mock_search.call_args_list[0][1]["where"]
        assert "machine" in normal_where
        assert "$in" in normal_where["machine"]
        assert "cnc-mill" in normal_where["machine"]["$in"]
        assert "CNC Mill" in normal_where["machine"]["$in"]

        # 2nd call: safety retrieval pre-filter
        safety_where = mock_search.call_args_list[1][1]["where"]
        assert "$and" in safety_where
        safety_machine_filter = safety_where["$and"][0]["machine"]
        assert "$in" in safety_machine_filter
        assert "cnc-mill" in safety_machine_filter["$in"]


def test_retrieval_no_cross_machine_fallback_when_empty():
    """Verify when a specific machine has no matches, it returns empty and does not fall back to other machines."""
    with patch("src.retrieval.search_documents") as mock_search, \
         patch("src.retrieval.create_embedding", return_value=[0.1] * 3072):

        # ChromaDB returns empty when filtering by non-existent machine
        mock_search.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        chunks = retrieve_documents(query="bearing failure", machine="Unmatched-Machine-999", n_results=5)
        assert chunks == []


# ============================================================
# TASK 3: Collision-Resistant Chunk IDs, Re-ingestion, & Deletion
# ============================================================

def test_same_filename_different_machines_no_collision():
    """Verify two documents with the identical filename for different machines produce distinct doc_id and chunk_ids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "safety.txt"
        file_path.write_text("General safety procedures: lockout electrical disconnect before maintenance.", encoding="utf-8")

        chunks_mill = create_chunks(
            file_path=file_path,
            document_type="safety",
            machine="CNC Mill",
        )

        chunks_press = create_chunks(
            file_path=file_path,
            document_type="safety",
            machine="Hydraulic Press",
        )

        doc_id_mill = chunks_mill[0]["metadata"]["doc_id"]
        doc_id_press = chunks_press[0]["metadata"]["doc_id"]

        assert doc_id_mill != doc_id_press
        assert "cnc-mill" in doc_id_mill
        assert "hydraulic-press" in doc_id_press

        chunk_id_mill = chunks_mill[0]["metadata"]["chunk_id"]
        chunk_id_press = chunks_press[0]["metadata"]["chunk_id"]
        assert chunk_id_mill != chunk_id_press


def test_sqlite_documents_same_filename_allowed():
    """Verify SQLite documents table allows two documents with the same filename for different machines."""
    initialize_database()

    id1 = save_document(
        filename="operating_guide.txt",
        document_type="manual",
        machine="CNC Mill",
        chunks=3,
    )

    id2 = save_document(
        filename="operating_guide.txt",
        document_type="manual",
        machine="Hydraulic Press",
        chunks=5,
    )

    docs = get_documents(limit=100)
    matched = [d for d in docs if d["filename"] == "operating_guide.txt"]
    assert len(matched) >= 2
    machines = {d["machine"] for d in matched}
    assert "cnc-mill" in machines
    assert "hydraulic-press" in machines

    # Cleanup
    delete_document(doc_id=matched[0]["doc_id"])
    delete_document(doc_id=matched[1]["doc_id"])


def test_reingestion_cleans_old_chunks():
    """Verify re-ingesting a document purges old chunks to avoid orphans when chunk count decreases."""
    doc_id = "test_reingest_doc_id"
    col = get_collection()

    # Index 3 chunks initially
    initial_chunks = [
        {"text": f"Chunk {i}", "metadata": {"doc_id": doc_id, "chunk_id": f"{doc_id}_c{i}", "machine": "cnc-mill"}}
        for i in range(1, 4)
    ]
    initial_embeddings = [[0.05] * 3072 for _ in range(3)]
    add_documents(chunks=initial_chunks, embeddings=initial_embeddings)

    res = col.get(where={"doc_id": doc_id})
    assert len(res["ids"]) == 3

    # Re-ingest with only 1 chunk
    updated_chunks = [
        {"text": "Only single chunk now", "metadata": {"doc_id": doc_id, "chunk_id": f"{doc_id}_c1", "machine": "cnc-mill"}}
    ]
    updated_embeddings = [[0.05] * 3072]
    add_documents(chunks=updated_chunks, embeddings=updated_embeddings, clean_existing=True)

    res_after = col.get(where={"doc_id": doc_id})
    assert len(res_after["ids"]) == 1
    assert res_after["ids"][0] == f"{doc_id}_c1"

    # Clean up
    delete_document_chunks(doc_id=doc_id)


def test_targeted_document_deletion():
    """Verify deleting one document deletes its chunks without affecting other documents."""
    col = get_collection()
    doc_a = "doc_alpha_unique"
    doc_b = "doc_beta_unique"

    chunks_a = [{"text": "Alpha", "metadata": {"doc_id": doc_a, "chunk_id": f"{doc_a}_c1", "machine": "cnc-mill"}}]
    chunks_b = [{"text": "Beta", "metadata": {"doc_id": doc_b, "chunk_id": f"{doc_b}_c1", "machine": "cnc-mill"}}]

    add_documents(chunks=chunks_a, embeddings=[[0.01] * 3072])
    add_documents(chunks=chunks_b, embeddings=[[0.02] * 3072])

    save_document(filename="alpha.txt", document_type="manual", machine="cnc-mill", chunks=1, doc_id=doc_a)
    save_document(filename="beta.txt", document_type="manual", machine="cnc-mill", chunks=1, doc_id=doc_b)

    # Delete doc_a only
    deleted = delete_document(doc_id=doc_a)
    assert deleted is True

    # doc_a chunks gone
    res_a = col.get(where={"doc_id": doc_a})
    assert len(res_a["ids"]) == 0

    # doc_b chunks remain
    res_b = col.get(where={"doc_id": doc_b})
    assert len(res_b["ids"]) == 1

    # Cleanup doc_b
    delete_document(doc_id=doc_b)


# ============================================================
# TASK 4: Embedding Distance Calibration Tests
# ============================================================

def test_distance_calibration_mathematical_identity():
    """Verify mathematical identity between squared L2 distance and cosine similarity for unit vectors."""
    # D^2 = 2 * (1 - cos_sim) => cos_sim = 1 - (D^2 / 2)
    def l2_squared_to_cosine(dist_sq: float) -> float:
        return 1.0 - (dist_sq / 2.0)

    # Threshold 0.70 corresponds to cos_sim = 0.65
    cos_at_threshold = l2_squared_to_cosine(0.70)
    assert abs(cos_at_threshold - 0.65) < 1e-6

    # Zero distance corresponds to cos_sim = 1.0
    assert l2_squared_to_cosine(0.0) == 1.0

    # Distance 2.0 corresponds to orthogonal (cos_sim = 0.0)
    assert l2_squared_to_cosine(2.0) == 0.0


def test_safety_evidence_distance_threshold_enforcement():
    """Verify safety evidence validation enforces calibrated distance threshold."""
    valid_text = "Before servicing electrical enclosures, verify main breaker lockout and tagout procedure."
    query = "Technician smells electrical smoke and sees sparking behind the motor drive."

    # Distance 0.40 (cos ~0.80) is accepted
    relevant_close_chunk = {
        "text": valid_text,
        "metadata": {"document_type": "safety", "machine": "cnc-mill"},
        "distance": 0.40,
    }
    assert is_safety_chunk_relevant_to_hazard(relevant_close_chunk, query=query) is True

    # Distance 0.75 (cos ~0.625) is REJECTED even with relevant text
    relevant_far_chunk = {
        "text": valid_text,
        "metadata": {"document_type": "safety", "machine": "cnc-mill"},
        "distance": 0.75,
    }
    assert is_safety_chunk_relevant_to_hazard(relevant_far_chunk, query=query) is False

    # Distance 0.30 but IRRELEVANT domain is REJECTED (fail-safe)
    irrelevant_text = "Always clean chip tray and wipe oil drips to prevent slip hazards on the floor."
    irrelevant_close_chunk = {
        "text": irrelevant_text,
        "metadata": {"document_type": "safety", "machine": "cnc-mill"},
        "distance": 0.30,
    }
    assert is_safety_chunk_relevant_to_hazard(irrelevant_close_chunk, query=query) is False


# ============================================================
# Phase 1 Auth & Supervisor Deletion Endpoint Regression
# ============================================================

def test_delete_document_endpoint_authorization(supervisor_client, technician_client):
    """Verify DELETE /documents/{doc_id} requires supervisor role."""
    # Technician forbidden
    resp_tech = technician_client.delete("/documents/any_doc_id")
    assert resp_tech.status_code == 403

    # Unauthenticated forbidden
    unauth_client = TestClient(app)
    resp_unauth = unauth_client.delete("/documents/any_doc_id")
    assert resp_unauth.status_code in {401, 403}

    # Supervisor non-existent gives 404
    resp_sup = supervisor_client.delete("/documents/nonexistent_doc_12345")
    assert resp_sup.status_code == 404


def test_deletion_does_not_purge_same_filename_on_other_machine():
    """Verify deleting safety.txt on Machine A does not delete safety.txt chunks on Machine B."""
    col = get_collection()
    doc_mill = "safety_cnc-mill_safety.txt"
    doc_press = "safety_hydraulic-press_safety.txt"

    chunks_mill = [
        {"text": "Mill safety rules", "metadata": {"doc_id": doc_mill, "chunk_id": f"{doc_mill}_c1", "machine": "cnc-mill", "document": "safety.txt"}}
    ]
    chunks_press = [
        {"text": "Press safety rules", "metadata": {"doc_id": doc_press, "chunk_id": f"{doc_press}_c1", "machine": "hydraulic-press", "document": "safety.txt"}}
    ]

    add_documents(chunks=chunks_mill, embeddings=[[0.01] * 3072])
    add_documents(chunks=chunks_press, embeddings=[[0.02] * 3072])

    save_document(filename="safety.txt", document_type="safety", machine="cnc-mill", chunks=1, doc_id=doc_mill)
    save_document(filename="safety.txt", document_type="safety", machine="hydraulic-press", chunks=1, doc_id=doc_press)

    # Delete doc_mill by doc_id
    deleted = delete_document(doc_id=doc_mill)
    assert deleted is True

    # Mill chunks and record are deleted
    res_mill = col.get(where={"doc_id": doc_mill})
    assert len(res_mill["ids"]) == 0

    # Press chunks MUST REMAIN INTACT!
    res_press = col.get(where={"doc_id": doc_press})
    assert len(res_press["ids"]) == 1
    assert res_press["ids"][0] == f"{doc_press}_c1"

    # Press SQLite record must remain
    docs = get_documents(limit=100)
    assert any(d["doc_id"] == doc_press for d in docs)

    # Cleanup doc_press
    delete_document(doc_id=doc_press)


def test_legacy_cnc_lathe_alias_retrieval():
    """Verify legacy ChromaDB metadata value 'CNC lathe' is included in retrieval filters and retrievable."""
    aliases = get_machine_retrieval_aliases("cnc-lathe")
    assert "CNC lathe" in aliases

    col = get_collection()
    res = col.get(where={"machine": {"$in": aliases}})
    # Stored collection contains 4 legacy chunks with machine == 'CNC lathe' and 16 with 'CNC Lathe'
    assert len(res["ids"]) >= 4
    stored_machines = {m["machine"] for m in res["metadatas"]}
    assert "CNC lathe" in stored_machines


def test_ambiguous_filename_lookup():
    """Verify looking up ambiguous filename across multiple machines returns ambiguity notice unless disambiguated."""
    initialize_database()
    col = get_collection()
    doc_id_a = "manual_cnc-mill_ambig_protocol.txt"
    doc_id_b = "manual_hydraulic-press_ambig_protocol.txt"

    add_documents(
        chunks=[{"text": "Mill protocol text", "metadata": {"doc_id": doc_id_a, "chunk_id": f"{doc_id_a}_c1", "machine": "cnc-mill", "document": "ambig_protocol.txt"}}],
        embeddings=[[0.01] * 3072]
    )
    add_documents(
        chunks=[{"text": "Press protocol text", "metadata": {"doc_id": doc_id_b, "chunk_id": f"{doc_id_b}_c1", "machine": "hydraulic-press", "document": "ambig_protocol.txt"}}],
        embeddings=[[0.02] * 3072]
    )

    save_document(filename="ambig_protocol.txt", document_type="manual", machine="cnc-mill", chunks=1, doc_id=doc_id_a)
    save_document(filename="ambig_protocol.txt", document_type="manual", machine="hydraulic-press", chunks=1, doc_id=doc_id_b)

    # Filename-only lookup without machine or doc_id returns clear ambiguity message
    res_ambig = get_document_content("ambig_protocol.txt")
    assert "Ambiguous document lookup" in res_ambig

    # Lookup by exact doc_id resolves unambiguously
    res_doc_a = get_document_content(doc_id_a)
    assert "Mill protocol text" in res_doc_a
    assert "Press protocol text" not in res_doc_a

    # Lookup by filename with machine disambiguation resolves
    res_mach_b = get_document_content("ambig_protocol.txt", machine="Hydraulic Press")
    assert "Press protocol text" in res_mach_b

    # Cleanup
    delete_document(doc_id=doc_id_a)
    delete_document(doc_id=doc_id_b)

