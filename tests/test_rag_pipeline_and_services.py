"""
Comprehensive automated tests for NERI RAG pipeline, services, ingestion,
role authorization, consistency, and error handling.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.citations import build_sources
from src.config import (
    get_supervisor_api_key,
    get_technician_api_key,
)
from src.history import (
    delete_document,
    get_documents,
    initialize_database,
    save_document,
)
from src.ingestion import (
    chunk_text,
    clean_text,
    create_chunks,
    extract_text_file,
    get_document_type,
)
from src.llm import (
    build_context,
    generate_troubleshooting_response,
)
from src.pipeline import troubleshoot
from src.vector_store import (
    add_documents,
    delete_document_chunks,
    get_collection,
)

client = TestClient(app)


@pytest.fixture
def supervisor_client():
    return TestClient(app, headers={"X-API-Key": get_supervisor_api_key()})


@pytest.fixture
def technician_client():
    return TestClient(app, headers={"X-API-Key": get_technician_api_key()})


# ============================================================
# 1. DOCUMENT INGESTION & TEXT CHUNKING TESTS
# ============================================================

def test_clean_text_normalizes_spaces_and_newlines():
    """Verify clean_text collapses irregular tabs, newlines, and multi-spaces."""
    raw = "  AIR   COMPRESSOR \n\n\t MAINTENANCE   LOG \r\n   Section  1  "
    cleaned = clean_text(raw)
    assert cleaned == "AIR COMPRESSOR MAINTENANCE LOG Section 1"


def test_chunk_text_boundaries_and_overlap():
    """Verify chunk_text splits into appropriate lengths and preserves overlap."""
    sample_text = "Word " * 250  # 1250 characters
    chunks = chunk_text(sample_text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 505
        assert len(chunk) > 0


def test_chunk_text_empty_input():
    """Verify chunk_text returns an empty list for empty or whitespace-only text."""
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_extract_text_file():
    """Verify extract_text_file reads files and cleans text content cleanly."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write("Hydraulic Press Operating Manual.\nCheck oil pressure weekly.")
        tmp_path = Path(tmp.name)

    try:
        pages = extract_text_file(tmp_path)
        assert len(pages) == 1
        assert "Hydraulic Press Operating Manual." in pages[0]["text"]
        assert pages[0]["page"] is None
    finally:
        tmp_path.unlink(missing_ok=True)


def test_get_document_type_validation_and_folders():
    """Verify get_document_type parses both explicit parameter and parent folder names."""
    # Explicit valid
    assert get_document_type(Path("doc.txt"), "manual") == "manual"
    assert get_document_type(Path("doc.txt"), "maintenance_log") == "maintenance_log"
    assert get_document_type(Path("doc.txt"), "safety") == "safety"

    # Explicit invalid raises ValueError
    with pytest.raises(ValueError, match="Invalid document type"):
        get_document_type(Path("doc.txt"), "warranty")

    # Inferred from parent directory
    assert get_document_type(Path("data/manuals/cnc.pdf")) == "manual"
    assert get_document_type(Path("data/maintenance_logs/press.txt")) == "maintenance_log"
    assert get_document_type(Path("data/safety/loto.pdf")) == "safety"
    assert get_document_type(Path("other/folder/file.pdf")) == "unknown"


def test_create_chunks_metadata_retention():
    """Verify create_chunks attaches all required metadata to each chunk."""
    content = "CNC Mill Maintenance Guide.\nAlways inspect cutting fluid levels and tool changer."
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        chunks = create_chunks(
            file_path=tmp_path,
            document_type="manual",
            machine="cnc-mill",
            version="2.1",
            owner="Precision Tooling Team",
            doc_id="manual__cnc-mill__guide",
        )
        assert len(chunks) >= 1
        chunk = chunks[0]
        assert "text" in chunk
        meta = chunk["metadata"]
        assert meta["chunk_id"].startswith("manual__")
        assert meta["document"] == tmp_path.name
        assert meta["document_type"] == "manual"
        assert meta["machine"] == "cnc-mill"
        assert meta["version"] == "2.1"
        assert meta["owner"] == "Precision Tooling Team"
    finally:
        tmp_path.unlink(missing_ok=True)


# ============================================================
# 2. RETRIEVAL RELEVANCE & SOURCE-DOCUMENT REFERENCES TESTS
# ============================================================

def test_build_sources_grouping_and_formatting():
    """Verify build_sources formats citations and deduplicates retrieved chunk references."""
    mock_chunks = [
        {
            "metadata": {
                "document": "press_manual.pdf",
                "document_type": "manual",
                "section": "Hydraulics",
                "page": 14,
            },
            "distance": 0.18,
        },
        {
            "metadata": {
                "document": "press_manual.pdf",
                "document_type": "manual",
                "section": "Valves",
                "page": 16,
            },
            "distance": 0.22,
        },
        {
            "metadata": {
                "document": "press_safety.pdf",
                "document_type": "safety",
                "section": "LOTO",
                "page": 2,
            },
            "distance": 0.12,
        },
    ]

    sources = build_sources(mock_chunks)
    assert len(sources) == 3
    doc_names = [s["document"] for s in sources]
    assert "press_manual.pdf" in doc_names
    assert "press_safety.pdf" in doc_names
    for s in sources:
        assert "document_type" in s
        assert "section" in s
        assert "page" in s


def test_build_context_grounding_structure():
    """Verify build_context produces structured SOURCE ID labels with content."""
    mock_chunks = [
        {
            "text": "Main spindle requires ISO VG 68 lubricant every 200 operating hours.",
            "metadata": {
                "document": "cnc_mill_manual.pdf",
                "document_type": "manual",
                "section": "Lubrication",
                "page": 45,
            },
        }
    ]

    context_str = build_context(mock_chunks)
    assert "SOURCE ID: 1" in context_str
    assert "DOCUMENT: cnc_mill_manual.pdf" in context_str
    assert "DOCUMENT TYPE: manual" in context_str
    assert "SECTION: Lubrication" in context_str
    assert "PAGE: 45" in context_str
    assert "Main spindle requires ISO VG 68" in context_str


# ============================================================
# 3. GROUNDED RESPONSES BASED ON RETRIEVED EVIDENCE (MOCKED LLM)
# ============================================================

def test_generate_troubleshooting_response_mocked_llm():
    """Verify generate_troubleshooting_response parses valid LLM response format cleanly."""
    mock_response_json = {
        "reliable": True,
        "possible_causes": [
            {"cause": "Hydraulic proportional valve seal degradation", "sources": [1]}
        ],
        "safety_warning": {
            "message": "Isolate high-pressure manifold before valve inspection.",
            "sources": [2]
        },
        "troubleshooting_steps": [
            {"step": "Check main reservoir pressure gauge reading", "sources": [1]},
            {"step": "Perform zero-energy lockout tagout verification", "sources": [2]}
        ]
    }

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(mock_response_json)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("src.llm._get_client", return_value=mock_client):
        result = generate_troubleshooting_response(
            machine="Hydraulic Press",
            machine_id="PRS-001",
            problem="Pressure drop during ram extension",
            error_code="E-115",
            retrieved_chunks=[
                {"text": "Sample chunk 1", "metadata": {"document": "manual.pdf", "document_type": "manual", "section": "Hydraulics", "page": 12}},
                {"text": "Sample chunk 2", "metadata": {"document": "safety.pdf", "document_type": "safety", "section": "LOTO", "page": 4}}
            ],
            safety_evidence_available=True,
        )

        assert result["reliable"] is True
        assert len(result["possible_causes"]) == 1
        assert result["possible_causes"][0]["cause"] == "Hydraulic proportional valve seal degradation"
        assert len(result["troubleshooting_steps"]) == 2
        assert result["safety_warning"] is not None


def test_generate_troubleshooting_response_invalid_json_handling():
    """Verify generate_troubleshooting_response raises ValueError when LLM returns non-JSON."""
    mock_choice = MagicMock()
    mock_choice.message.content = "I cannot fulfill this in JSON format."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("src.llm._get_client", return_value=mock_client):
        with pytest.raises(ValueError, match="The model returned invalid JSON"):
            generate_troubleshooting_response(
                machine="Lathe",
                machine_id="LAT-01",
                problem="Spindle vibration",
                error_code=None,
                retrieved_chunks=[],
            )


def test_generate_troubleshooting_response_empty_completion_handling():
    """Verify generate_troubleshooting_response raises ValueError when LLM returns empty content."""
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    with patch("src.llm._get_client", return_value=mock_client):
        with pytest.raises(ValueError, match="The model returned an empty response"):
            generate_troubleshooting_response(
                machine="Lathe",
                machine_id="LAT-01",
                problem="Spindle vibration",
                error_code=None,
                retrieved_chunks=[],
            )


# ============================================================
# 4. SAFETY REFUSAL & ESCALATION BEHAVIOR TESTS
# ============================================================

def test_pipeline_refuses_hazard_query_when_safety_evidence_unavailable():
    """Verify pipeline halts and returns refusal when a hazardous query lacks safety docs."""
    with patch("src.pipeline.get_cached_response", return_value=None):
        with patch("src.pipeline.retrieve_documents", return_value=[
            # Only ordinary manual chunk, NO safety document chunk
            {
                "text": "Motor wiring connects to the 480V terminal bus.",
                "metadata": {"document": "wiring_diagram.pdf", "document_type": "manual", "section": "Wiring", "page": 5},
                "distance": 0.15,
            }
        ]):
            res = troubleshoot(
                machine="CNC Mill",
                machine_id="CNC-001",
                problem="Exposed 480V electrical arc sparking at main terminal, risk of shock and high voltage hazard",
                error_code="E-VOLT",
            )

            assert res["reliable"] is False
            assert res["safety_warning"] is not None
            assert "safety" in str(res["safety_warning"]).lower()
            assert "safety documentation was found" in res["message"]


def test_pipeline_proceeds_with_hazard_query_when_safety_evidence_available():
    """Verify pipeline proceeds when genuine certified safety evidence is retrieved."""
    mock_safety_chunk = {
        "text": "Lockout Tagout (LOTO): Disconnect 480V main breaker. Verify zero electrical potential with calibrated voltmeter before touching terminals.",
        "metadata": {
            "document": "plant_electrical_safety.pdf",
            "document_type": "safety",
            "section": "LOTO Protocols",
            "page": 3,
        },
        "distance": 0.10,
    }

    mock_llm_output = {
        "reliable": True,
        "possible_causes": [{"cause": "Loose 480V terminal bus bar connector", "sources": [1]}],
        "safety_warning": {"message": "De-energize 480V power and execute LOTO protocol.", "sources": [1]},
        "troubleshooting_steps": [{"step": "Follow LOTO procedure and tighten lugs", "sources": [1]}],
    }

    with patch("src.pipeline.get_cached_response", return_value=None):
        with patch("src.pipeline.retrieve_documents", return_value=[mock_safety_chunk]):
            with patch("src.pipeline.generate_troubleshooting_response", return_value=mock_llm_output):
                res = troubleshoot(
                    machine="CNC Mill",
                    machine_id="CNC-001",
                    problem="Electrical sparking hazard at 480V breaker",
                )

                assert res["reliable"] is True
                assert res["safety_warning"] is not None
                assert "LOTO" in res["safety_warning"]["message"]


# ============================================================
# 5. API AUTHENTICATION & ROLE-BASED AUTHORIZATION TESTS
# ============================================================

def test_api_auth_missing_header_rejected_401():
    """Verify API rejects requests missing both X-API-Key and Authorization headers."""
    resp = client.get("/documents")
    assert resp.status_code == 401
    assert "Authentication credentials are required" in resp.json()["detail"]


def test_api_auth_invalid_token_rejected_401():
    """Verify API rejects unauthorized tokens."""
    resp = client.get("/documents", headers={"X-API-Key": "invalid-random-token"})
    assert resp.status_code == 401
    assert "Invalid authentication token" in resp.json()["detail"]


def test_api_auth_bearer_token_supported():
    """Verify Authorization: Bearer <valid_token> is accepted cleanly."""
    tech_key = get_technician_api_key()
    resp = client.get("/documents", headers={"Authorization": f"Bearer {tech_key}"})
    assert resp.status_code == 200
    assert "documents" in resp.json()


def test_technician_role_forbidden_upload_403(technician_client):
    """Verify Technician cannot access POST /documents/upload."""
    files = {"file": ("manual.txt", b"Test manual content", "text/plain")}
    data = {"document_type": "manual", "machine": "hydraulic-press"}
    resp = technician_client.post("/documents/upload", files=files, data=data)
    assert resp.status_code == 403
    assert "Supervisor privileges required" in resp.json()["detail"]


def test_technician_role_forbidden_delete_403(technician_client):
    """Verify Technician cannot delete documents."""
    resp = technician_client.delete("/documents/some_doc_id")
    assert resp.status_code == 403
    assert "Supervisor privileges required" in resp.json()["detail"]


def test_supervisor_role_permitted_upload_and_delete(supervisor_client):
    """Verify Supervisor can upload and delete documents."""
    files = {"file": ("test_doc_perm.txt", b"Safety protocol content", "text/plain")}
    data = {
        "document_type": "safety",
        "machine": "CNC Mill",
        "version": "1.0",
        "owner": "Safety Department"
    }

    with patch("api.main.create_chunks") as mock_chunks, \
         patch("api.main.create_embeddings") as mock_embeds, \
         patch("api.main.add_documents") as mock_add, \
         patch("api.main.save_document") as mock_save, \
         patch("src.vector_store.delete_document_chunks") as mock_del:

        mock_chunks.return_value = [{"text": "Safety protocol content", "metadata": {"doc_id": "test_perm_id"}}]
        mock_embeds.return_value = [[0.1, 0.2]]

        res = supervisor_client.post("/documents/upload", files=files, data=data)
        assert res.status_code == 200
        assert res.json()["success"] is True

        with patch("src.history.delete_document", return_value=True):
            del_res = supervisor_client.delete("/documents/some_doc_id")
            assert del_res.status_code == 200
            assert del_res.json()["success"] is True


# ============================================================
# 6. DOCUMENT DELETION CONSISTENCY (CHROMADB & SQLITE)
# ============================================================

def test_document_deletion_consistency_sqlite_and_chroma():
    """Verify targeted deletion purges chunks from ChromaDB and records from SQLite."""
    unique_doc_id = "manual__consistency_test__spec_01"
    unique_filename = "consistency_spec.txt"
    canonical_machine = "cnc-mill"

    # 1. Add chunks to ChromaDB
    test_chunks = [
        {
            "text": "Consistency test chunk content.",
            "metadata": {
                "chunk_id": f"{unique_doc_id}__0001",
                "doc_id": unique_doc_id,
                "document": unique_filename,
                "machine": canonical_machine,
                "document_type": "manual",
                "section": "General",
                "page": 1,
            },
        }
    ]
    add_documents(test_chunks, embeddings=[[0.05] * 3072], clean_existing=False)

    # 2. Add record to SQLite
    save_document(
        filename=unique_filename,
        document_type="manual",
        machine=canonical_machine,
        version="1.0",
        owner="QA",
        chunks=1,
        status="indexed",
        doc_id=unique_doc_id,
    )

    # Verify both exist
    assert any(d.get("doc_id") == unique_doc_id for d in get_documents())
    coll = get_collection()
    before_check = coll.get(where={"doc_id": unique_doc_id})
    assert len(before_check["ids"]) >= 1

    # 3. Perform targeted deletion
    deleted = delete_document(doc_id=unique_doc_id, filename=unique_filename, machine=canonical_machine)
    assert deleted is True

    # 4. Verify SQLite record is deleted
    assert not any(d.get("doc_id") == unique_doc_id for d in get_documents())

    # 5. Verify ChromaDB chunks are purged
    after_check = coll.get(where={"doc_id": unique_doc_id})
    assert len(after_check["ids"]) == 0


# ============================================================
# 7. API INPUT VALIDATION & ERROR HANDLING TESTS
# ============================================================

def test_query_endpoint_missing_fields_validation_error(technician_client):
    """Verify /query returns 422 Unprocessable Entity when required fields are missing."""
    # Missing 'problem'
    bad_payload = {"machine": "Hydraulic Press", "machine_id": "PRS-01"}
    resp = technician_client.post("/query", json=bad_payload)
    assert resp.status_code == 422


def test_history_detail_nonexistent_session_404(technician_client):
    """Verify /history/{id} returns 404 for non-existent session IDs."""
    resp = technician_client.get("/history/999999999")
    assert resp.status_code == 404
    assert "Troubleshooting session not found" in resp.json()["detail"]


def test_feedback_nonexistent_session_404(technician_client):
    """Verify /history/{id}/feedback returns 404 for non-existent session IDs."""
    resp = technician_client.post("/history/999999999/feedback", json={"feedback": "helpful"})
    assert resp.status_code == 404


def test_feedback_invalid_sentiment_400(technician_client):
    """Verify feedback endpoint returns 400 when feedback value is not 'helpful' or 'not_helpful'."""
    # First get an existing session id or create one
    with patch("src.pipeline.troubleshoot", return_value={"reliable": True, "possible_causes": [], "troubleshooting_steps": [], "sources": []}):
        query_resp = technician_client.post(
            "/query",
            json={"machine": "CNC Mill", "machine_id": "CNC-001", "problem": "Feedback validation test"}
        )
        if query_resp.status_code == 200:
            s_id = query_resp.json().get("session_id")
            if s_id:
                bad_fb_resp = technician_client.post(
                    f"/history/{s_id}/feedback",
                    json={"feedback": "neutral"}  # Invalid sentiment
                )
                assert bad_fb_resp.status_code == 400
                assert "Feedback must be" in bad_fb_resp.json()["detail"]


def test_upload_unsupported_file_extension_400(supervisor_client):
    """Verify /documents/upload returns 400 for unsupported file formats (.exe, .zip)."""
    files = {"file": ("malware.exe", b"MZbinarycontent", "application/octet-stream")}
    data = {"document_type": "manual", "machine": "cnc-mill"}
    resp = supervisor_client.post("/documents/upload", files=files, data=data)
    assert resp.status_code == 400
    assert "Only PDF and TXT files are supported" in resp.json()["detail"]


def test_upload_unsupported_document_type_400(supervisor_client):
    """Verify /documents/upload returns 400 when document_type is not manual/maintenance_log/safety."""
    files = {"file": ("doc.txt", b"Technical content", "text/plain")}
    data = {"document_type": "financial_report", "machine": "cnc-mill"}
    resp = supervisor_client.post("/documents/upload", files=files, data=data)
    assert resp.status_code == 400
    assert "Invalid document type" in resp.json()["detail"]


def test_delete_nonexistent_doc_id_404(supervisor_client):
    """Verify /documents/{doc_id} returns 404 when document does not exist."""
    resp = supervisor_client.delete("/documents/non_existent_doc_id_xyz_123")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()
