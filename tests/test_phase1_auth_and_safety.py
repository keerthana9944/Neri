import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app
from src.config import (
    get_supervisor_api_key,
    get_technician_api_key,
    validate_config,
)
from src.guardrails import (
    is_safety_critical,
    has_safety_evidence,
    build_safety_evidence_response,
)
from src.pipeline import troubleshoot


client = TestClient(app)


# ============================================================
# TASK A: Authentication & Configuration Hardening Tests
# ============================================================

def test_validate_config_fails_when_supervisor_key_missing():
    """Verify validate_config fails closed if NERI_SUPERVISOR_KEY is missing or empty."""
    with patch.dict(os.environ, {"NERI_SUPERVISOR_KEY": "", "NERI_TECHNICIAN_KEY": "tech-key-1"}):
        with pytest.raises(ValueError, match="NERI_SUPERVISOR_KEY is missing or empty"):
            validate_config()


def test_validate_config_fails_when_technician_key_missing():
    """Verify validate_config fails closed if NERI_TECHNICIAN_KEY is missing or empty."""
    with patch.dict(os.environ, {"NERI_SUPERVISOR_KEY": "super-key-1", "NERI_TECHNICIAN_KEY": ""}):
        with pytest.raises(ValueError, match="NERI_TECHNICIAN_KEY is missing or empty"):
            validate_config()


def test_validate_config_fails_when_keys_are_identical():
    """Verify validate_config fails closed if both keys are configured with the same value."""
    with patch.dict(os.environ, {"NERI_SUPERVISOR_KEY": "same-key-123", "NERI_TECHNICIAN_KEY": "same-key-123"}):
        with pytest.raises(ValueError, match="must not be identical"):
            validate_config()


def test_validate_config_succeeds_with_valid_distinct_keys():
    """Verify validate_config passes when distinct keys are configured."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "valid-api-key",
        "NERI_SUPERVISOR_KEY": "super-unique-key",
        "NERI_TECHNICIAN_KEY": "tech-unique-key"
    }):
        assert validate_config() is True


def test_api_fails_closed_when_server_keys_misconfigured():
    """Verify API returns HTTP 500 when server authentication keys are missing or identical."""
    with patch("api.main.get_supervisor_api_key", return_value="identical-key"), \
         patch("api.main.get_technician_api_key", return_value="identical-key"):
        res = client.get("/history", headers={"X-API-Key": "identical-key"})
        assert res.status_code == 500
        assert "server authentication configuration is invalid" in res.json()["detail"].lower()


def test_unauthenticated_api_requests_rejected():
    """Verify that sensitive API endpoints reject unauthenticated requests (HTTP 401)."""
    # Query endpoint
    res = client.post("/query", json={
        "machine": "CNC Mill",
        "machine_id": "CNC-001",
        "problem": "spindle noise"
    })
    assert res.status_code == 401
    assert "credentials are required" in res.json()["detail"].lower()

    # Empty token header
    res = client.get("/history", headers={"X-API-Key": "   "})
    assert res.status_code == 401

    # Malformed Bearer token
    res = client.get("/history", headers={"Authorization": "Bearer   "})
    assert res.status_code == 401

    # History endpoint
    res = client.get("/history")
    assert res.status_code == 401

    # Documents endpoint
    res = client.get("/documents")
    assert res.status_code == 401

    # Document upload endpoint
    res = client.post(
        "/documents/upload",
        data={"document_type": "manual"}
    )
    assert res.status_code == 401


def test_technician_cannot_upload_documents():
    """Verify that authenticated technicians are forbidden from uploading documents (HTTP 403)."""
    tech_key = get_technician_api_key()
    headers = {"X-API-Key": tech_key}
    files = {"file": ("test.txt", b"test content", "text/plain")}
    data = {"document_type": "manual"}

    res = client.post("/documents/upload", headers=headers, files=files, data=data)
    assert res.status_code == 403
    assert "supervisor privileges required" in res.json()["detail"].lower()


def test_supervisor_can_upload_documents():
    """Verify that authorized Supervisors can upload documents."""
    super_key = get_supervisor_api_key()
    headers = {"X-API-Key": super_key}
    files = {"file": ("test_doc.txt", b"Safety protocol content", "text/plain")}
    data = {
        "document_type": "safety",
        "machine": "CNC Mill",
        "version": "1.0",
        "owner": "Safety Department"
    }

    with patch("api.main.create_chunks") as mock_chunks, \
         patch("api.main.create_embeddings") as mock_embeds, \
         patch("api.main.add_documents") as mock_add, \
         patch("api.main.save_document") as mock_save:

        mock_chunks.return_value = [{"text": "Safety protocol content", "metadata": {}}]
        mock_embeds.return_value = [[0.1, 0.2]]

        res = client.post("/documents/upload", headers=headers, files=files, data=data)
        assert res.status_code == 200
        assert res.json()["success"] is True
        assert res.json()["status"] == "indexed"


def test_authenticated_technician_can_access_permitted_endpoints():
    """Verify authenticated technicians can query and read history."""
    tech_key = get_technician_api_key()
    headers = {"X-API-Key": tech_key}

    with patch("api.main.get_history") as mock_get_history:
        mock_get_history.return_value = [{"id": 1, "machine": "CNC Mill"}]
        res = client.get("/history", headers=headers)
        assert res.status_code == 200
        assert "sessions" in res.json()


def test_invalid_api_key_rejected():
    """Verify requests with invalid API keys return HTTP 401."""
    headers = {"X-API-Key": "invalid-secret-key-12345"}
    res = client.get("/history", headers=headers)
    assert res.status_code == 401
    assert "invalid" in res.json()["detail"].lower()


def test_streamlit_apptest_privilege_escalation_prevention():
    """
    Test using streamlit.testing.v1.AppTest against the live Streamlit app
    that ?role=supervisor and ?view=documents cannot authenticate a session
    or grant Supervisor privileges.
    """
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.query_params["role"] = "supervisor"
    at.query_params["view"] = "documents"
    at.run()

    # Verify session state remains unauthenticated
    assert at.session_state["logged_in"] is False
    assert at.session_state["user_role"] is None
    assert at.session_state["username"] is None
    assert at.session_state["show_landing_page"] is True


# ============================================================
# TASK B: Safety Guardrails & Evidence Relevance Tests
# ============================================================

def test_is_safety_critical_hazard_phrasings():
    """Test detection of explicit and alternative safety hazard wordings."""
    hazardous_queries = [
        "High voltage cable sparking near motor",
        "Visible smoke and burning smell coming from electrical panel",
        "Electric shock when touching enclosure",
        "Arc flash observed in cabinet",
        "Pressurized hydraulic line rupture with fluid spraying",
        "Toxic gas fumes leaking from valve",
        "Chemical leak on floor near operator station",
        "Crush hazard: blade guard loose and entrapment risk",
        "Fire hazard detected near heater",
    ]

    for q in hazardous_queries:
        assert is_safety_critical(q) is True, f"Failed to detect hazard in: '{q}'"


def test_is_safety_critical_ordinary_queries():
    """Test that ordinary operational queries are NOT marked as safety-critical."""
    ordinary_queries = [
        "Spindle speed fluctuating during milling",
        "Low coolant level warning on display",
        "Abnormal hum noise during heavy cut",
        "Poor surface finish on aluminum part",
        "Slow table movement on X-axis",
        "Tool wear after 500 cycles",
        "Pressure fluctuation in lubrication pump",
    ]

    for q in ordinary_queries:
        assert is_safety_critical(q) is False, f"False positive hazard detected in: '{q}'"


def test_has_safety_evidence_relevant_evidence():
    """Verify that genuinely relevant safety documentation is accepted."""
    query = "Electrical spark near spindle motor cabinet"
    relevant_safety_chunk = [{
        "text": "Electrical Safety Protocol: Disconnect main 480V breaker and lockout power before accessing cabinet wiring.",
        "metadata": {"document_type": "safety", "document": "safety_elec.pdf"}
    }]
    assert has_safety_evidence(relevant_safety_chunk, query=query) is True


def test_has_safety_evidence_irrelevant_evidence_rejected():
    """
    Verify that unrelated safety documents (e.g. hearing protection, footwear)
    cannot authorize an electrical hazard troubleshooting request.
    """
    query = "Electrical spark and arc flash inside main control cabinet"
    irrelevant_safety_chunk = [{
        "text": "Hearing protection policy: All operators must wear 30dB earplugs in Bay 2 due to loud machining noise.",
        "metadata": {"document_type": "safety", "document": "safety_ppe.pdf"}
    }]
    assert has_safety_evidence(irrelevant_safety_chunk, query=query) is False

    footwear_safety_chunk = [{
        "text": "Footwear standard: Wear steel-toe oil-resistant boots to prevent slip hazards on plant walkways.",
        "metadata": {"document_type": "safety", "document": "safety_footwear.pdf"}
    }]
    assert has_safety_evidence(footwear_safety_chunk, query=query) is False


def test_has_safety_evidence_distant_vector_match_rejected():
    """Verify that chunks with high vector distance (> 0.70) are rejected."""
    query = "High voltage arc flash"
    distant_chunk = [{
        "text": "Electrical safety disconnect and circuit breaker lockout procedures.",
        "metadata": {"document_type": "safety"},
        "distance": 0.82
    }]
    assert has_safety_evidence(distant_chunk, query=query) is False


def test_has_safety_evidence_missing_or_empty_rejected():
    """Verify empty or non-safety chunks are rejected."""
    query = "Electrical spark"
    assert has_safety_evidence([], query=query) is False

    manual_chunk = [{"text": "General maintenance guide", "metadata": {"document_type": "manual"}}]
    assert has_safety_evidence(manual_chunk, query=query) is False

    empty_safety = [{"text": "", "metadata": {"document_type": "safety"}}]
    assert has_safety_evidence(empty_safety, query=query) is False


def test_hazardous_query_with_irrelevant_safety_doc_fails_safely():
    """
    Verify that when retrieval only finds an IRRELEVANT safety document,
    the pipeline fails safely: does not generate ordinary steps, and returns refusal.
    """
    with patch("src.pipeline.get_cached_response", return_value=None), \
         patch("src.pipeline.retrieve_documents") as mock_retrieve, \
         patch("src.pipeline.cache_response"):

        # Retrieval returns an irrelevant safety document (earplugs) for an electrical hazard
        mock_retrieve.return_value = [
            {
                "text": "Hearing protection policy: Operators must wear approved earplugs in the machining area.",
                "metadata": {"document_type": "safety", "document": "ppe_hearing.pdf", "section": "1.0", "page": 1, "chunk_id": "h1"}
            },
            {
                "text": "Check motor bearings for mechanical vibration.",
                "metadata": {"document_type": "manual", "document": "manual.pdf", "section": "2.0", "page": 10, "chunk_id": "m1"}
            }
        ]

        res = troubleshoot(
            machine="CNC Mill",
            machine_id="CNC-001",
            problem="Arc flash and visible electrical sparking in cabinet",
            error_code=None
        )

        assert res["reliable"] is False
        assert "safety-critical condition" in res["message"].lower()
        assert res["troubleshooting_steps"] == []
        assert res["safety_warning"] is not None


def test_safety_warnings_never_silently_stripped():
    """
    Verify that when relevant safety documentation is available and LLM generates a safety warning,
    the warning is NEVER silently removed.
    """
    with patch("src.pipeline.get_cached_response", return_value=None), \
         patch("src.pipeline.retrieve_documents") as mock_retrieve, \
         patch("src.pipeline.generate_troubleshooting_response") as mock_llm, \
         patch("src.pipeline.cache_response"):

        mock_retrieve.return_value = [
            {
                "text": "Electrical Lockout Procedure: Disconnect 480V power and tag circuit breaker before opening cabinet.",
                "metadata": {"document_type": "safety", "document": "safety.pdf", "section": "1.0", "page": 2, "chunk_id": "s1"}
            }
        ]

        mock_llm.return_value = {
            "reliable": True,
            "possible_causes": [{"cause": "Loose ground wire", "sources": [1]}],
            "safety_warning": "Disconnect main power before opening electrical box.",
            "troubleshooting_steps": [{"step": "Inspect ground wire", "sources": [1]}]
        }

        res = troubleshoot(
            machine="CNC Mill",
            machine_id="CNC-001",
            problem="Arc flash in electrical cabinet",
            error_code=None
        )

        assert res["safety_warning"] == "Disconnect main power before opening electrical box."


# ============================================================
# TASK C: Configuration & Startup Import Regression Tests
# ============================================================

def test_config_module_exports_authentication_functions():
    """Verify src.config exports get_supervisor_api_key and get_technician_api_key."""
    import src.config as cfg
    assert hasattr(cfg, "get_supervisor_api_key"), "src.config missing get_supervisor_api_key"
    assert hasattr(cfg, "get_technician_api_key"), "src.config missing get_technician_api_key"
    assert callable(cfg.get_supervisor_api_key)
    assert callable(cfg.get_technician_api_key)
    assert isinstance(cfg.get_supervisor_api_key(), str)
    assert isinstance(cfg.get_technician_api_key(), str)


def test_src_package_reexports_auth_functions():
    """Verify src package namespace exports authentication functions."""
    import src
    assert hasattr(src, "get_supervisor_api_key")
    assert hasattr(src, "get_technician_api_key")


def test_streamlit_app_imports_and_startup_regression():
    """Verify app/streamlit_app.py imports its configuration dependencies and starts up without ImportError."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert not at.exception, f"Streamlit app startup raised exceptions: {at.exception}"
