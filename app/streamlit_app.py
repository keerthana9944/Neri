import html
import re
import sys
from pathlib import Path
from typing import Literal
import requests
import streamlit as st

# ==================================================
# Project Setup & Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from src.config import (
    RAG_API_URL,
    get_supervisor_api_key,
    get_technician_api_key,
)
from src.history import auto_seed_if_empty

# Ensure all files under data/ are automatically seeded & indexed on startup
try:
    auto_seed_if_empty()
except Exception as e:
    print(f"Auto-seed note: {e}")


def get_api_headers():
    """Return authentication headers for API requests based on session state."""
    role = st.session_state.get("user_role")
    if role == "Supervisor":
        key = get_supervisor_api_key()
        return {"X-API-Key": key} if key else {}
    elif role == "Technician":
        key = get_technician_api_key()
        return {"X-API-Key": key} if key else {}
    return {}


# Helper functions for API calls with direct Python fallback (Streamlit Cloud & local support)
def fetch_history_sessions():
    try:
        resp = requests.get(f"{RAG_API_URL}/history", headers=get_api_headers(), timeout=2)
        if resp.status_code == 200:
            return resp.json().get("sessions", [])
    except Exception:
        pass
    from src.history import get_history
    return get_history()


def fetch_documents_list():
    try:
        resp = requests.get(f"{RAG_API_URL}/documents", headers=get_api_headers(), timeout=2)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    from src.history import get_documents
    return get_documents()


def fetch_session_detail(session_id: int):
    try:
        resp = requests.get(f"{RAG_API_URL}/history/{session_id}", headers=get_api_headers(), timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    from src.history import get_session
    return get_session(session_id)


def post_feedback(session_id: int, feedback: str, comment: str | None = None):
    try:
        resp = requests.post(
            f"{RAG_API_URL}/history/{session_id}/feedback",
            json={"feedback": feedback, "comment": comment},
            headers=get_api_headers(),
            timeout=2
        )
        if resp.status_code == 200:
            return True
    except Exception:
        pass
    from src.history import save_feedback
    return save_feedback(session_id=session_id, feedback=feedback, feedback_comment=comment)


def post_troubleshoot_query(payload: dict):
    try:
        resp = requests.post(f"{RAG_API_URL}/query", json=payload, headers=get_api_headers(), timeout=60)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    from src.pipeline import troubleshoot
    from src.history import save_session
    response = troubleshoot(
        machine=payload["machine"],
        machine_id=payload["machine_id"],
        problem=payload["problem"],
        error_code=payload.get("error_code")
    )
    session_id = save_session(
        machine=payload["machine"],
        machine_id=payload["machine_id"],
        problem=payload["problem"],
        error_code=payload.get("error_code"),
        response=response
    )
    response["session_id"] = session_id
    return response


def post_document_upload(
    file_name: str,
    file_bytes: bytes,
    document_type: str,
    machine: str | None = None,
    version: str | None = None,
    owner: str | None = None,
):
    if st.session_state.get("user_role") != "Supervisor":
        raise PermissionError("Supervisor privileges required to upload documents.")

    try:
        files = {"file": (file_name, file_bytes)}
        data = {
            "document_type": document_type,
            "machine": machine,
            "version": version,
            "owner": owner
        }
        res = requests.post(f"{RAG_API_URL}/documents/upload", files=files, data=data, headers=get_api_headers(), timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass

    from src.config import UPLOAD_DIR
    from src.ingestion import create_chunks
    from src.embeddings import create_embeddings
    from src.vector_store import add_documents, delete_document_chunks
    from src.history import save_document
    from src.machine_norm import generate_document_id, normalize_machine_id

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / Path(file_name).name
    file_path.write_bytes(file_bytes)

    canonical_machine = normalize_machine_id(machine)
    doc_id = generate_document_id(
        document_type=document_type,
        machine=canonical_machine,
        filename=file_name,
    )

    delete_document_chunks(
        doc_id=doc_id,
        document_name=file_name,
        machine=canonical_machine,
    )

    chunks = create_chunks(
        file_path,
        document_type=document_type,
        machine=canonical_machine,
        version=version,
        owner=owner,
        doc_id=doc_id,
    )
    if not chunks:
        raise ValueError("No readable text was found in the document.")

    embeddings = create_embeddings([c["text"] for c in chunks])
    add_documents(chunks=chunks, embeddings=embeddings, clean_existing=False)
    save_document(
        filename=file_name,
        document_type=document_type,
        machine=canonical_machine,
        version=version,
        owner=owner,
        chunks=len(chunks),
        status="indexed",
        doc_id=doc_id,
    )
    return {
        "success": True,
        "message": "Document uploaded and indexed successfully.",
        "doc_id": doc_id,
        "document": file_name,
        "document_type": document_type,
        "machine": canonical_machine,
        "version": version,
        "owner": owner,
        "chunks": len(chunks),
        "status": "indexed",
    }


def post_document_delete(
    doc_id: str | None = None,
    filename: str | None = None,
    machine: str | None = None,
):
    if st.session_state.get("user_role") != "Supervisor":
        raise PermissionError("Supervisor privileges required to delete documents.")
    try:
        if doc_id:
            resp = requests.delete(f"{RAG_API_URL}/documents/{doc_id}", headers=get_api_headers(), timeout=5)
            if resp.status_code == 200:
                return resp.json().get("success", False)
    except Exception:
        pass
    from src.history import delete_document
    return delete_document(doc_id=doc_id, filename=filename, machine=machine)


def fetch_document_content(identifier: str, machine: str | None = None) -> str:
    try:
        from src.history import get_document_content
        content = get_document_content(identifier, machine=machine)
        if content and content != "Document content unavailable.":
            return content
    except Exception:
        pass
    try:
        params = {"machine": machine} if machine else {}
        resp = requests.get(f"{RAG_API_URL}/documents/{identifier}/content", params=params, headers=get_api_headers(), timeout=2)
        if resp.status_code == 200:
            return resp.json().get("content", "")
    except Exception:
        pass
    return "Document content unavailable."


def clean_html(html_str: str) -> str:
    """Removes leading and trailing whitespace from each line to prevent Streamlit Markdown from treating HTML as code blocks."""
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "".join(lines)


def format_document_preview_html(doc_text: str, is_expanded: bool = False) -> str:
    """Formats document content preview into a highly readable, semantic HTML card."""
    if not doc_text or doc_text.strip() == "" or doc_text.strip() == "Document content unavailable.":
        return clean_html(
            '<div style="background: #F8F3FC; border: 1.5px dashed #DDB7FF; border-radius: 12px; '
            'padding: 28px 20px; text-align: center; color: #5A7383; font-size: 0.875rem; margin-top: 6px;">'
            '<div style="font-weight: 700; color: #1B4558; margin-bottom: 4px;">No Content Preview Available</div>'
            '<div>The text content for this document could not be retrieved or is empty.</div>'
            '</div>'
        )

    escaped = html.escape(doc_text.strip())
    raw_paras = [p.strip() for p in re.split(r'\n{2,}', escaped) if p.strip()]
    formatted_paras = []
    for p in raw_paras:
        lines = [line.strip() for line in p.split('\n') if line.strip()]
        if len(lines) == 1 and (lines[0].isupper() or lines[0].endswith(':')) and len(lines[0]) <= 80:
            formatted_paras.append(
                f'<h4 style="font-size: 0.95rem; font-weight: 800; color: #1B4558; '
                f'margin: 16px 0 6px 0; letter-spacing: 0.2px; border-bottom: 1px solid #F3E8FF; padding-bottom: 4px;">'
                f'{lines[0]}</h4>'
            )
        else:
            p_content = "<br>".join(lines)
            formatted_paras.append(
                f'<p style="margin: 0 0 10px 0; color: #1B4558; font-size: 0.875rem; line-height: 1.65; word-break: break-word;">'
                f'{p_content}</p>'
            )

    body_html = "".join(formatted_paras)
    max_height_style = "max-height: none;" if is_expanded else "max-height: 320px; overflow-y: auto;"

    return clean_html(
        f'<div style="background: #FFFFFF; border: 1.5px solid #E5E7EB; border-radius: 12px; '
        f'padding: 20px 22px; {max_height_style} box-shadow: inset 0 1px 3px rgba(27,69,88,0.03); '
        f'transition: all 0.2s ease;">'
        f'{body_html}'
        f'</div>'
    )



st.set_page_config(
    page_title="Neri — AI Maintenance Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Session State & URL State Initialization (Refresh-Safe)
# ==================================================

query_view = st.query_params.get("view")
query_role = st.query_params.get("role")
query_page = st.query_params.get("page")

valid_views = {
    "dashboard": "Dashboard",
    "troubleshoot": "Troubleshoot",
    "history": "History",
    "documents": "Documents"
}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["show_landing_page"] = query_page != "login"
    st.session_state["user_role"] = None
    st.session_state["username"] = None

# Only align navigation view for already authenticated active sessions
if st.session_state.get("logged_in"):
    if query_view and query_view.lower() in valid_views:
        st.session_state["main_navigation_radio"] = valid_views[query_view.lower()]

if "show_landing_page" not in st.session_state:
    st.session_state["show_landing_page"] = query_page != "login"

if "user_role" not in st.session_state:
    st.session_state["user_role"] = None

if "username" not in st.session_state:
    st.session_state["username"] = None

if "troubleshooting_result" not in st.session_state:
    st.session_state["troubleshooting_result"] = None

if "troubleshooting_input" not in st.session_state:
    st.session_state["troubleshooting_input"] = {}

if "history_detail" not in st.session_state:
    st.session_state["history_detail"] = None

if "show_feedback_comment" not in st.session_state:
    st.session_state["show_feedback_comment"] = False

if "history_filter_machine" not in st.session_state:
    st.session_state["history_filter_machine"] = None

if "login_selected_role" not in st.session_state:
    st.session_state["login_selected_role"] = "Technician"

# ==================================================
# SVG Vector Icon Helper System
# ==================================================

def get_svg_icon(name: str, color: str = "currentColor", size: int = 18) -> str:
    """Generates clean, professional monochrome vector SVG icons."""
    icons = {
        "logo": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><circle cx="12" cy="12" r="3"/></svg>',
        "shield": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "shield-check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
        "alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "document": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "wrench": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
        "history": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "dashboard": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="15" width="7" height="6"/></svg>',
        "user": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "logout": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>',
        "arrow-right": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "server": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>',
        "clock": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "file-text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "trash": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>',
    }
    return icons.get(name, f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>')


def get_neri_mascot_svg(width=220, height=260) -> str:
    """Generates the official NERI robot mascot artwork in the new color palette."""
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 240 300" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="robotBodyGrad" x1="50%" y1="0%" x2="50%" y2="100%">
          <stop offset="0%" stop-color="#FFFFFF" />
          <stop offset="70%" stop-color="#F8F3FC" />
          <stop offset="100%" stop-color="#EAD9FA" />
        </linearGradient>
        <linearGradient id="robotHeadGrad" x1="20%" y1="0%" x2="80%" y2="100%">
          <stop offset="0%" stop-color="#FFFFFF" />
          <stop offset="75%" stop-color="#FAF5FF" />
          <stop offset="100%" stop-color="#E9D5FF" />
        </linearGradient>
        <radialGradient id="redBallGrad" cx="35%" cy="35%" r="65%">
          <stop offset="0%" stop-color="#FF6B6A" />
          <stop offset="60%" stop-color="#EF3837" />
          <stop offset="100%" stop-color="#B91C1C" />
        </radialGradient>
        <filter id="robotShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#1B4558" flood-opacity="0.12" />
        </filter>
      </defs>

      <!-- Antenna Sparkle lines -->
      <path d="M 32 75 L 20 66" stroke="#EF3837" stroke-width="3" stroke-linecap="round" />
      <path d="M 26 94 L 12 92" stroke="#EF3837" stroke-width="3" stroke-linecap="round" />

      <!-- Robot Group -->
      <g filter="url(#robotShadow)">
        <!-- Antenna Left -->
        <line x1="72" y1="75" x2="52" y2="38" stroke="#EF3837" stroke-width="4.5" stroke-linecap="round" />
        <circle cx="48" cy="32" r="9.5" fill="url(#redBallGrad)" />

        <!-- Antenna Right -->
        <line x1="135" y1="72" x2="155" y2="34" stroke="#EF3837" stroke-width="4.5" stroke-linecap="round" />
        <circle cx="159" cy="28" r="9.5" fill="url(#redBallGrad)" />

        <!-- Waving Hand -->
        <path d="M 48 160 C 30 145 28 115 48 95 C 56 88 70 98 63 112 C 55 125 55 150 62 165 Z" fill="url(#robotBodyGrad)" stroke="#E5E7EB" stroke-width="1.5" />
        <circle cx="45" cy="100" r="16" fill="url(#robotBodyGrad)" stroke="#E5E7EB" stroke-width="1.5" />

        <!-- Motion Lines -->
        <path d="M 16 122 C 10 130 8 140 10 148" stroke="#EF3837" stroke-width="3" stroke-linecap="round" />
        <path d="M 23 118 C 18 126 16 134 18 142" stroke="#EF3837" stroke-width="3" stroke-linecap="round" />

        <!-- Head -->
        <rect x="46" y="65" width="118" height="92" rx="38" fill="url(#robotHeadGrad)" stroke="#FFFFFF" stroke-width="2.5" />
        
        <!-- Face Screen -->
        <rect x="62" y="80" width="86" height="58" rx="20" fill="#1B4558" />
        
        <!-- Eyes -->
        <ellipse cx="86" cy="106" rx="7" ry="9" fill="#FFFFFF" />
        <ellipse cx="124" cy="106" rx="7" ry="9" fill="#FFFFFF" />
        
        <!-- Mouth -->
        <path d="M 98 115 Q 105 123 112 115" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" fill="none" />
        
        <!-- Cheeks -->
        <ellipse cx="75" cy="115" rx="6.5" ry="4" fill="#DDB7FF" opacity="0.8" />
        <ellipse cx="135" cy="115" rx="6.5" ry="4" fill="#DDB7FF" opacity="0.8" />

        <!-- Body -->
        <path d="M 58 150 C 58 150 44 245 105 245 C 166 245 152 150 152 150 Z" fill="url(#robotBodyGrad)" stroke="#FFFFFF" stroke-width="1.5" />
        
        <!-- Chest Brand N -->
        <text x="96" y="196" font-family="'Outfit', 'Plus Jakarta Sans', sans-serif" font-weight="800" font-size="28" fill="#EF3837">N</text>

        <!-- Right Arm -->
        <path d="M 148 160 C 165 174 170 198 156 220 C 150 206 145 182 145 164 Z" fill="url(#robotBodyGrad)" stroke="#E5E7EB" stroke-width="1.5" />
      </g>
    </svg>"""


# ==================================================
# NERI DESIGN SYSTEM CSS — MANDATORY COLOR PALETTE
# ==================================================

st.markdown(
    clean_html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

        :root {
            --primary-red: #EF3837;
            --primary-red-hover: #D92D2C;
            --primary-red-light: #FEE2E2;
            --secondary-lavender: #DDB7FF;
            --secondary-lavender-light: #F3E8FF;
            --page-bg: #F8F3FC;
            --card-surface: #FFFFFF;
            --text-primary: #1B4558;
            --text-secondary: #5A7383;
            --border-color: #E5E7EB;
            --active-nav-bg: #F3E8FF;
            --warning-bg: #FFFBEB;
            --warning-border: #FCD34D;
            --warning-text: #92400E;
            --hazard-bg: #FEF2F2;
            --hazard-border: #FCA5A5;
            --hazard-text: #991B1B;
            --success-bg: #ECFDF5;
            --success-border: #A7F3D0;
            --success-text: #065F46;
            color-scheme: light !important;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
        }

        h1, h2, h3, h4, .neri-card-header {
            font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
            color: var(--text-primary) !important;
        }

        .stApp {
            background-color: var(--page-bg) !important;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1240px;
        }

        /* Streamlit Primary Buttons — Kalvium Red #EF3837 */
        div.stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: var(--primary-red) !important;
            background-color: var(--primary-red) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: 1px solid var(--primary-red) !important;
            font-weight: 700 !important;
            padding: 10px 22px !important;
            box-shadow: 0 4px 14px rgba(239, 56, 55, 0.25) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background: var(--primary-red-hover) !important;
            background-color: var(--primary-red-hover) !important;
            border-color: var(--primary-red-hover) !important;
            box-shadow: 0 6px 18px rgba(239, 56, 55, 0.35) !important;
            transform: translateY(-1px);
        }

        div.stButton > button[kind="primary"]:active,
        button[data-testid="baseButton-primary"]:active {
            transform: scale(0.985) !important;
            box-shadow: 0 2px 8px rgba(239, 56, 55, 0.2) !important;
        }

        /* Streamlit Secondary Buttons */
        div.stButton > button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {
            background-color: var(--card-surface) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover {
            border-color: var(--secondary-lavender) !important;
            background-color: #FAF5FF !important;
            color: var(--primary-red) !important;
            transform: translateY(-1px);
        }

        div.stButton > button[kind="secondary"]:active,
        button[data-testid="baseButton-secondary"]:active {
            transform: scale(0.985) !important;
        }

        /* Streamlit Form Inputs, Selectboxes, File Uploaders & TextAreas */
        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            font-size: 0.875rem !important;
        }

        div[data-baseweb="input"],
        div[data-baseweb="base-input"],
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        textarea[data-testid="stTextArea"],
        section[data-testid="stFileUploaderDropzone"],
        div[data-testid="stFileUploaderDropzone"],
        div[data-baseweb="file-uploader"] {
            background-color: var(--card-surface) !important;
            border: 1.5px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            box-shadow: 0 1px 4px rgba(27, 69, 88, 0.04) !important;
            transition: all 0.2s ease !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        textarea[data-testid="stTextArea"] {
            color: var(--text-primary) !important;
            background-color: var(--card-surface) !important;
            font-weight: 500 !important;
        }

        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="base-input"]:focus-within,
        div[data-baseweb="select"]:focus-within,
        textarea[data-testid="stTextArea"]:focus {
            border-color: var(--primary-red) !important;
            box-shadow: 0 0 0 3px rgba(239, 56, 55, 0.15) !important;
        }

        /* Segmented Control / Horizontal Filter Boxes */
        div[data-testid="stSegmentedControl"],
        div.stSegmentedControl,
        div.st-key-doc_type_filter {
            margin-top: 1px;
        }

        div[data-testid="stSegmentedControl"] button,
        div.stSegmentedControl button,
        div.st-key-doc_type_filter button,
        div.st-key-doc_type_filter [data-baseweb="button-group"] button {
            background-color: var(--card-surface) !important;
            color: var(--text-primary) !important;
            border: 1.5px solid var(--border-color) !important;
            border-radius: 9px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 3px rgba(27, 69, 88, 0.04) !important;
        }

        div[data-testid="stSegmentedControl"] button:hover,
        div.stSegmentedControl button:hover,
        div.st-key-doc_type_filter button:hover {
            border-color: var(--primary-red) !important;
            background-color: var(--page-bg) !important;
            color: var(--primary-red) !important;
        }

        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[aria-selected="true"],
        div.stSegmentedControl button[aria-checked="true"],
        div.stSegmentedControl button[aria-selected="true"],
        div.st-key-doc_type_filter button[aria-checked="true"],
        div.st-key-doc_type_filter button[aria-selected="true"],
        div.st-key-doc_type_filter button[data-selected="true"] {
            background-color: var(--active-nav-bg) !important;
            border-color: var(--primary-red) !important;
            color: var(--primary-red) !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(239, 56, 55, 0.18) !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99 !important;
        }

        #MainMenu, footer {
            visibility: hidden !important;
            height: 0px !important;
        }

        /* Cards & Panels */
        .neri-card {
            background-color: var(--card-surface);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 10px rgba(27, 69, 88, 0.04);
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease;
        }

        .neri-card:hover {
            box-shadow: 0 8px 24px rgba(27, 69, 88, 0.08);
            border-color: var(--secondary-lavender);
            transform: translateY(-2px);
        }

        .neri-card-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Safety Warning Box (Amber Panel) */
        .warning-box {
            padding: 18px 20px;
            border-radius: 12px;
            border: 1.5px solid var(--warning-border);
            background-color: var(--warning-bg);
            color: var(--warning-text);
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(217, 119, 6, 0.08);
        }

        .warning-box-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--warning-text);
        }

        /* Safety Critical Refusal Box (Red Panel) */
        .critical-box {
            padding: 20px 22px;
            border-radius: 12px;
            border: 1.5px solid var(--hazard-border);
            background-color: var(--hazard-bg);
            color: var(--hazard-text);
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(239, 56, 55, 0.08);
        }

        .critical-box-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--primary-red);
        }

        .success-box {
            padding: 18px 20px;
            border-radius: 12px;
            border: 1.5px solid var(--success-border);
            background-color: var(--success-bg);
            color: var(--success-text);
            margin-bottom: 18px;
        }

        .source-box {
            padding: 16px 18px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background-color: var(--card-surface);
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }

        .source-box:hover {
            border-color: var(--secondary-lavender);
            transform: translateX(3px);
        }

        .document-box {
            padding: 18px 20px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background-color: var(--card-surface);
            margin-bottom: 12px;
            box-shadow: 0 1px 4px rgba(27, 69, 88, 0.03);
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.2s ease;
        }

        .document-box:hover {
            border-color: var(--secondary-lavender);
            box-shadow: 0 4px 14px rgba(221, 183, 255, 0.18);
            transform: translateY(-1px);
        }

        /* Numbered Badges & Workflow Steps */
        .neri-step-item {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 14px 0;
            border-bottom: 1px solid #F3F4F6;
            transition: all 0.2s ease;
        }

        .neri-step-item:last-child {
            border-bottom: none;
        }

        .neri-badge-num {
            background-color: var(--primary-red);
            color: #FFFFFF;
            font-size: 0.85rem;
            font-weight: 800;
            min-width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 2px 6px rgba(239, 56, 55, 0.25);
        }

        /* Status Pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.775rem;
            font-weight: 700;
            background-color: var(--success-bg);
            color: var(--success-text);
            border: 1px solid var(--success-border);
        }

        .status-pill.active {
            background-color: var(--secondary-lavender-light);
            color: #6B21A8;
            border: 1px solid var(--secondary-lavender);
        }

        .status-pill.warning {
            background-color: var(--warning-bg);
            color: var(--warning-text);
            border: 1px solid var(--warning-border);
        }

        .status-pill.error {
            background-color: var(--hazard-bg);
            color: var(--primary-red);
            border: 1px solid var(--hazard-border);
        }

        /* Expanders */
        div[data-testid="stExpander"],
        .streamlit-expander {
            background-color: var(--card-surface) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 1px 4px rgba(27, 69, 88, 0.03) !important;
            overflow: hidden !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stExpander"]:hover {
            border-color: var(--secondary-lavender) !important;
        }

        div[data-testid="stExpander"] details summary,
        div[data-testid="stExpander"] [data-testid="stExpanderSummary"],
        .streamlit-expanderHeader {
            background-color: #FAFAFC !important;
            border-bottom: 1px solid #F3F4F6 !important;
            padding: 12px 16px !important;
        }

        div[data-testid="stExpander"] details summary:hover,
        div[data-testid="stExpander"] [data-testid="stExpanderSummary"]:hover {
            background-color: #F8F3FC !important;
        }

        div[data-testid="stExpander"] details summary *,
        div[data-testid="stExpander"] [data-testid="stExpanderSummary"] * {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            font-size: 0.925rem !important;
        }

        /* Micro-Animations: Page Entrance & Smooth Transitions */
        @keyframes neriFadeIn {
            from {
                opacity: 0;
                transform: translateY(4px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .stApp, .block-container, [data-testid="stMain"] {
            animation: neriFadeIn 0.22s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Sleek Modern Scrollbars */
        ::-webkit-scrollbar {
            width: 7px;
            height: 7px;
        }
        ::-webkit-scrollbar-track {
            background: var(--page-bg);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--secondary-lavender);
            border-radius: 6px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #C492FA;
        }

        /* Responsive Layout Polish & Mobile Optimizations */
        @media (max-width: 992px) {
            .block-container {
                padding-top: 1.25rem !important;
                padding-bottom: 2.5rem !important;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 2rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            div[data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 12px !important;
            }

            div[data-testid="column"]:empty {
                display: none !important;
                margin: 0 !important;
                padding: 0 !important;
                height: 0 !important;
            }

            div.stButton > button,
            button[data-testid="baseButton-primary"],
            button[data-testid="baseButton-secondary"] {
                min-height: 44px !important;
                width: 100% !important;
            }

            div[data-testid="stSegmentedControl"],
            div.stSegmentedControl,
            div.st-key-doc_type_filter {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 6px !important;
            }

            div[data-testid="stSegmentedControl"] button,
            div.stSegmentedControl button,
            div.st-key-doc_type_filter button {
                flex: 1 1 calc(50% - 6px) !important;
                text-align: center !important;
                min-height: 42px !important;
            }

            .neri-card {
                padding: 18px !important;
                border-radius: 12px !important;
            }

            .warning-box, .critical-box {
                padding: 16px !important;
            }
        }

        @media (max-width: 480px) {
            div[data-testid="stSegmentedControl"] button,
            div.stSegmentedControl button,
            div.st-key-doc_type_filter button {
                flex: 1 1 100% !important;
            }
        }

        /* Accessibility: reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                transform: none !important;
            }
        }
    </style>
    """),
    unsafe_allow_html=True,
)

# Dynamic layout CSS injection based on authentication state
is_user_logged_in = st.session_state.get("logged_in", False)

if is_user_logged_in:
    dynamic_layout_css = """
    <style>
        @media (min-width: 769px) {
            section[data-testid="stSidebar"],
            section[aria-label="sidebar"] {
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
                transform: none !important;
                margin-left: 0 !important;
                left: 0 !important;
                top: 0 !important;
                bottom: 0 !important;
                height: 100vh !important;
                width: 260px !important;
                min-width: 260px !important;
                max-width: 260px !important;
                background: #FFFFFF !important;
                border-right: 1px solid #E5E7EB !important;
                box-sizing: border-box !important;
                z-index: 99999 !important;
                position: fixed !important;
            }

            [data-testid="stMain"],
            section.main,
            div.main {
                margin-left: 260px !important;
                width: calc(100% - 260px) !important;
            }
        }

        @media (max-width: 768px) {
            section[data-testid="stSidebar"],
            section[aria-label="sidebar"] {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                bottom: 0 !important;
                height: 100vh !important;
                width: 280px !important;
                max-width: 85vw !important;
                background: #FFFFFF !important;
                border-right: 1px solid #E5E7EB !important;
                box-shadow: 4px 0 24px rgba(27, 69, 88, 0.15) !important;
                z-index: 999999 !important;
                box-sizing: border-box !important;
            }

            [data-testid="stMain"],
            section.main,
            div.main {
                margin-left: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
            }
        }

        .block-container {
            max-width: 1240px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
    </style>
    """
else:
    dynamic_layout_css = """
    <style>
        section[data-testid="stSidebar"],
        section[aria-label="sidebar"] {
            display: none !important;
            width: 0px !important;
            visibility: hidden !important;
            margin-left: -9999px !important;
        }

        [data-testid="stMain"],
        section.main,
        div.main {
            margin-left: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }

        .block-container {
            max-width: 1200px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
    </style>
    """

st.markdown(clean_html(dynamic_layout_css), unsafe_allow_html=True)


# ==================================================
# PUBLIC LANDING PAGE
# ==================================================

if not st.session_state["logged_in"] and st.session_state.get("show_landing_page", True):

    mascot_hero_svg = get_neri_mascot_svg(width=210, height=250)

    st.markdown(
        clean_html(f"""
        <style>
            .neri-nav-bar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 16px 28px;
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 16px;
                margin-bottom: 32px;
                box-shadow: 0 2px 12px rgba(27, 69, 88, 0.04);
            }}

            .neri-nav-brand {{
                display: flex;
                align-items: center;
                gap: 12px;
                text-decoration: none;
            }}

            .neri-brand-badge {{
                width: 38px;
                height: 38px;
                border-radius: 10px;
                background: #EF3837;
                color: #FFFFFF;
                font-size: 20px;
                font-weight: 800;
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: 'Outfit', sans-serif;
                box-shadow: 0 4px 12px rgba(239, 56, 55, 0.28);
            }}

            .neri-brand-title {{
                font-family: 'Outfit', sans-serif;
                font-size: 1.35rem;
                font-weight: 800;
                color: #1B4558;
                letter-spacing: -0.5px;
            }}

            .neri-nav-links {{
                display: flex;
                align-items: center;
                gap: 24px;
            }}

            .neri-nav-link {{
                color: #5A7383;
                text-decoration: none;
                font-size: 0.9rem;
                font-weight: 600;
                transition: color 0.2s;
            }}

            .neri-nav-link:hover {{
                color: #EF3837;
            }}

            .neri-nav-cta {{
                padding: 9px 20px;
                background: #EF3837;
                color: #FFFFFF !important;
                border-radius: 999px;
                text-decoration: none;
                font-weight: 700;
                font-size: 0.875rem;
                box-shadow: 0 4px 12px rgba(239, 56, 55, 0.25);
                transition: all 0.2s ease;
            }}

            .neri-nav-cta:hover {{
                background: #D92D2C;
                transform: translateY(-1px);
            }}

            /* HERO SECTION */
            .neri-hero-container {{
                display: grid;
                grid-template-columns: 1.4fr 1fr;
                gap: 36px;
                align-items: center;
                padding: 40px;
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 20px;
                margin-bottom: 36px;
                box-shadow: 0 4px 20px rgba(27, 69, 88, 0.05);
            }}

            .neri-hero-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 14px;
                background: #F3E8FF;
                border: 1px solid #DDB7FF;
                border-radius: 999px;
                font-size: 0.775rem;
                font-weight: 700;
                color: #6B21A8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 16px;
            }}

            .neri-hero-h1 {{
                font-size: 2.75rem;
                font-weight: 800;
                line-height: 1.15;
                color: #1B4558;
                letter-spacing: -1px;
                margin: 0 0 16px 0;
            }}

            .neri-hero-h1 span {{
                color: #EF3837;
            }}

            .neri-hero-desc {{
                font-size: 1.05rem;
                line-height: 1.6;
                color: #5A7383;
                margin: 0 0 28px 0;
            }}

            .neri-hero-actions {{
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 32px;
            }}

            .neri-btn-main {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 13px 28px;
                background: #EF3837;
                color: #FFFFFF !important;
                border-radius: 12px;
                text-decoration: none;
                font-weight: 700;
                font-size: 1rem;
                box-shadow: 0 6px 18px rgba(239, 56, 55, 0.28);
                transition: all 0.2s ease;
            }}

            .neri-btn-main:hover {{
                background: #D92D2C;
                transform: translateY(-2px);
                box-shadow: 0 8px 22px rgba(239, 56, 55, 0.35);
            }}

            .neri-hero-stats {{
                display: flex;
                gap: 28px;
                border-top: 1px solid #F3F4F6;
                padding-top: 20px;
            }}

            .neri-stat-box {{
                display: flex;
                flex-direction: column;
            }}

            .neri-stat-value {{
                font-family: 'Outfit', sans-serif;
                font-size: 1.4rem;
                font-weight: 800;
                color: #1B4558;
            }}

            .neri-stat-label {{
                font-size: 0.775rem;
                font-weight: 600;
                color: #5A7383;
            }}

            /* HERO MASCOT CARD (Dedicated Right Column) */
            .neri-mascot-frame {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: #F8F3FC;
                border: 1.5px solid #DDB7FF;
                border-radius: 18px;
                padding: 24px 20px;
                box-shadow: 0 8px 24px rgba(221, 183, 255, 0.16);
                text-align: center;
            }}

            .neri-mascot-caption {{
                margin-top: 12px;
            }}

            .neri-mascot-name {{
                font-size: 1.05rem;
                font-weight: 800;
                color: #1B4558;
            }}

            .neri-mascot-status {{
                display: inline-flex;
                align-items: center;
                gap: 5px;
                font-size: 0.75rem;
                font-weight: 700;
                color: #EF3837;
                background: #FEE2E2;
                padding: 2px 10px;
                border-radius: 999px;
                margin-top: 4px;
            }}

            /* FEATURES GRID */
            .neri-features-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 20px;
                margin-bottom: 36px;
            }}

            .neri-feature-card {{
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 16px;
                padding: 24px;
                transition: all 0.25s ease;
            }}

            .neri-feature-card:hover {{
                border-color: #DDB7FF;
                transform: translateY(-3px);
                box-shadow: 0 8px 24px rgba(27, 69, 88, 0.06);
            }}

            .neri-feature-icon {{
                width: 44px;
                height: 44px;
                border-radius: 12px;
                background: #F3E8FF;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 14px;
            }}

            .neri-feature-title {{
                font-size: 1.05rem;
                font-weight: 700;
                color: #1B4558;
                margin: 0 0 8px 0;
            }}

            .neri-feature-text {{
                font-size: 0.85rem;
                line-height: 1.5;
                color: #5A7383;
                margin: 0;
            }}

            /* ARCHITECTURE OVERVIEW */
            .neri-arch-banner {{
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 18px;
                padding: 28px 32px;
                margin-bottom: 36px;
            }}

            .neri-arch-steps {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-top: 18px;
            }}

            .neri-arch-step {{
                background: #F8F3FC;
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 16px;
            }}

            .neri-arch-step-num {{
                font-size: 0.725rem;
                font-weight: 800;
                color: #EF3837;
                text-transform: uppercase;
                margin-bottom: 4px;
            }}

            .neri-arch-step-name {{
                font-size: 0.925rem;
                font-weight: 700;
                color: #1B4558;
                margin-bottom: 4px;
            }}

            .neri-arch-step-detail {{
                font-size: 0.775rem;
                color: #5A7383;
                line-height: 1.4;
            }}

            /* FOOTER */
            .neri-landing-footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 0;
                border-top: 1px solid #E5E7EB;
                font-size: 0.825rem;
                color: #5A7383;
            }}

            @media (max-width: 960px) {{
                .neri-hero-container {{ grid-template-columns: 1fr; }}
                .neri-features-grid {{ grid-template-columns: repeat(2, 1fr); }}
                .neri-arch-steps {{ grid-template-columns: repeat(2, 1fr); }}
            }}
        </style>

        <div class="neri-nav-bar">
            <div class="neri-nav-brand">
                <div class="neri-brand-badge">N</div>
                <div>
                    <div class="neri-brand-title">NERI</div>
                    <div style="font-size: 0.65rem; font-weight: 700; color: #EF3837; letter-spacing: 1px; text-transform: uppercase;">Maintenance AI</div>
                </div>
            </div>
            <div class="neri-nav-links">
                <a href="#features" class="neri-nav-link">Capabilities</a>
                <a href="#architecture" class="neri-nav-link">Architecture</a>
                <a href="#safety" class="neri-nav-link">Safety Guardrails</a>
                <a href="?page=login" class="neri-nav-cta">Sign In →</a>
            </div>
        </div>

        <div class="neri-hero-container">
            <div>
                <div class="neri-hero-badge">
                    {get_svg_icon("cpu", color="#6B21A8", size=14)} INDUSTRIAL RAG TROUBLESHOOTING
                </div>
                <h1 class="neri-hero-h1">
                    Grounded AI diagnostics for <span>industrial machinery</span>
                </h1>
                <p class="neri-hero-desc">
                    Empower plant maintenance technicians with accurate, citation-backed troubleshooting guidance. Fully grounded in verified OEM manuals, maintenance history, and safety specifications.
                </p>
                <div class="neri-hero-actions">
                    <a href="?page=login" class="neri-btn-main">
                        Launch Troubleshooter {get_svg_icon("arrow-right", color="#FFFFFF", size=18)}
                    </a>
                </div>
                <div class="neri-hero-stats">
                    <div class="neri-stat-box">
                        <span class="neri-stat-value">100%</span>
                        <span class="neri-stat-label">Grounded OEM Citations</span>
                    </div>
                    <div class="neri-stat-box">
                        <span class="neri-stat-value">Fail-Closed</span>
                        <span class="neri-stat-label">LOTO Safety Protection</span>
                    </div>
                    <div class="neri-stat-box">
                        <span class="neri-stat-value">Sub-Second</span>
                        <span class="neri-stat-label">Vector Retrieval</span>
                    </div>
                </div>
            </div>
            <div class="neri-mascot-frame">
                {mascot_hero_svg}
                <div class="neri-mascot-caption">
                    <div class="neri-mascot-name">NERI Troubleshooting Co-Pilot</div>
                    <div class="neri-mascot-status">● Plant Intelligence Active</div>
                </div>
            </div>
        </div>

        <div id="features" style="margin-bottom: 12px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">CORE CAPABILITIES</div>
            <h3 style="font-size: 1.5rem; font-weight: 800; color: #1B4558; margin: 0 0 18px 0;">Built for mission-critical plant operations</h3>
        </div>

        <div class="neri-features-grid">
            <div class="neri-feature-card">
                <div class="neri-feature-icon">{get_svg_icon("document", color="#EF3837", size=22)}</div>
                <h4 class="neri-feature-title">Grounded RAG Retrieval</h4>
                <p class="neri-feature-text">Answers are synthesized strictly from approved manufacturer manuals, maintenance logs, and schematics.</p>
            </div>
            <div class="neri-feature-card">
                <div class="neri-feature-icon">{get_svg_icon("shield", color="#EF3837", size=22)}</div>
                <h4 class="neri-feature-title">Strict Safety Guardrails</h4>
                <p class="neri-feature-text">Hazard queries halt immediately if relevant safety evidence is missing. Strict Lockout/Tagout enforcement.</p>
            </div>
            <div class="neri-feature-card">
                <div class="neri-feature-icon">{get_svg_icon("wrench", color="#EF3837", size=22)}</div>
                <h4 class="neri-feature-title">Machine Normalization</h4>
                <p class="neri-feature-text">Canonical machine ID mapping handles equipment aliases safely across mills, presses, and lathes.</p>
            </div>
            <div class="neri-feature-card">
                <div class="neri-feature-icon">{get_svg_icon("history", color="#EF3837", size=22)}</div>
                <h4 class="neri-feature-title">Archival & Feedback</h4>
                <p class="neri-feature-text">Every diagnostic case is archived with technician feedback ratings to continuously refine maintenance knowledge.</p>
            </div>
        </div>

        <div id="architecture" class="neri-arch-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px;">SYSTEM ARCHITECTURE</div>
                    <h3 style="font-size: 1.25rem; font-weight: 800; color: #1B4558; margin: 2px 0 0 0;">Deterministic Pipeline with Fail-Closed Verification</h3>
                </div>
                <span class="status-pill active">{get_svg_icon("server", color="#6B21A8", size=14)} Enterprise Architecture</span>
            </div>
            <div class="neri-arch-steps">
                <div class="neri-arch-step">
                    <div class="neri-arch-step-num">Step 1 &bull; Ingestion</div>
                    <div class="neri-arch-step-name">Deterministic ID & Chunks</div>
                    <div class="neri-arch-step-detail">Uploads parsed and bound to stable document IDs and canonical machine aliases.</div>
                </div>
                <div class="neri-arch-step">
                    <div class="neri-arch-step-num">Step 2 &bull; Retrieval</div>
                    <div class="neri-arch-step-name">Calibrated Similarity</div>
                    <div class="neri-arch-step-detail">ChromaDB semantic search with cosine distance filtering and machine metadata scoping.</div>
                </div>
                <div class="neri-arch-step">
                    <div class="neri-arch-step-num">Step 3 &bull; Guardrails</div>
                    <div class="neri-arch-step-name">Hazard Verification</div>
                    <div class="neri-arch-step-detail">Safety classifier validates query against relevant safety procedures before proceeding.</div>
                </div>
                <div class="neri-arch-step">
                    <div class="neri-arch-step-num">Step 4 &bull; Response</div>
                    <div class="neri-arch-step-name">Structured Diagnostics</div>
                    <div class="neri-arch-step-detail">Verified causes, numbered steps, and exact OEM document citations delivered to technicians.</div>
                </div>
            </div>
        </div>

        <div class="neri-landing-footer">
            <div><strong>NERI</strong> &bull; AI-Powered Industrial Maintenance Troubleshooting Assistant</div>
            <div>Kalvium Industrial AI Submission &bull; Grounded in Approved Documentation</div>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.stop()


# ==================================================
# LOGIN PAGE
# ==================================================

if not st.session_state["logged_in"]:

    st.markdown(
        clean_html("""
        <style>
            .login-wrapper {
                max-width: 460px;
                margin: 40px auto 20px auto;
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 20px;
                padding: 38px 40px 32px;
                box-shadow: 0 10px 30px rgba(27, 69, 88, 0.08);
            }

            .login-brand-header {
                text-align: center;
                margin-bottom: 24px;
            }

            .login-brand-badge {
                width: 48px;
                height: 48px;
                border-radius: 12px;
                background: #EF3837;
                color: #FFFFFF;
                font-size: 26px;
                font-weight: 800;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-family: 'Outfit', sans-serif;
                box-shadow: 0 4px 14px rgba(239, 56, 55, 0.28);
                margin-bottom: 8px;
            }

            .login-brand-name {
                font-family: 'Outfit', sans-serif;
                font-size: 1.5rem;
                font-weight: 800;
                color: #1B4558;
                letter-spacing: -0.5px;
            }

            .login-brand-subtitle {
                font-size: 0.7rem;
                font-weight: 800;
                color: #EF3837;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-top: 2px;
            }

            .login-title-h1 {
                font-size: 1.75rem;
                font-weight: 800;
                color: #1B4558;
                margin: 0 0 6px 0;
                text-align: center;
            }

            .login-subtitle {
                font-size: 0.875rem;
                color: #5A7383;
                text-align: center;
                margin: 0 0 20px 0;
            }

            .login-credentials-note {
                background: #F8F3FC;
                border: 1px solid #DDB7FF;
                border-radius: 10px;
                padding: 12px 14px;
                font-size: 0.775rem;
                color: #1B4558;
                margin-top: 16px;
                line-height: 1.5;
            }
        </style>
        """),
        unsafe_allow_html=True
    )

    login_left, login_center, login_right = st.columns([1, 1.3, 1])

    with login_center:
        st.markdown(
            clean_html("""
            <div class="login-brand-header">
                <div class="login-brand-badge">N</div>
                <div class="login-brand-name">NERI</div>
                <div class="login-brand-subtitle">AI Maintenance Assistant</div>
            </div>
            <div class="login-title-h1">Sign In</div>
            <div class="login-subtitle">Select role and enter your plant credentials</div>
            """),
            unsafe_allow_html=True
        )

        selected_role = st.session_state.get("login_selected_role", "Technician")

        role_col1, role_col2 = st.columns(2)
        with role_col1:
            if st.button(
                "Technician",
                type="primary" if selected_role == "Technician" else "secondary",
                use_container_width=True,
                key="btn_role_tech"
            ):
                st.session_state["login_selected_role"] = "Technician"
                st.rerun()

        with role_col2:
            if st.button(
                "Supervisor",
                type="primary" if selected_role == "Supervisor" else "secondary",
                use_container_width=True,
                key="btn_role_super"
            ):
                st.session_state["login_selected_role"] = "Supervisor"
                st.rerun()

        with st.form("login_form"):
            default_user = "technician" if selected_role == "Technician" else "supervisor"
            username_input = st.text_input("Username", value=default_user, placeholder="Enter username")
            password_input = st.text_input("Password", type="password", placeholder="Enter password")

            submit_login = st.form_submit_button(
                f"Sign In as {selected_role} →",
                type="primary",
                use_container_width=True
            )

            if submit_login:
                u_clean = username_input.strip().lower()
                p_clean = password_input.strip()

                if u_clean == "technician" and p_clean == "tech123" and selected_role == "Technician":
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = "Technician"
                    st.session_state["username"] = "technician"
                    st.session_state["show_landing_page"] = False
                    st.session_state["main_navigation_radio"] = "Dashboard"
                    st.query_params.clear()
                    st.query_params["view"] = "dashboard"
                    st.query_params["role"] = "technician"
                    st.rerun()

                elif u_clean == "supervisor" and p_clean == "super123" and selected_role == "Supervisor":
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = "Supervisor"
                    st.session_state["username"] = "supervisor"
                    st.session_state["show_landing_page"] = False
                    st.session_state["main_navigation_radio"] = "Dashboard"
                    st.query_params.clear()
                    st.query_params["view"] = "dashboard"
                    st.query_params["role"] = "supervisor"
                    st.rerun()

                else:
                    st.error(f"Invalid credentials for {selected_role}. Use {default_user} / {'tech123' if selected_role == 'Technician' else 'super123'}.")

        st.markdown(
            clean_html("""
            <div class="login-credentials-note">
                <strong>Standard Credentials:</strong><br>
                &bull; Technician: <code>technician</code> / <code>tech123</code><br>
                &bull; Supervisor: <code>supervisor</code> / <code>super123</code>
            </div>
            """),
            unsafe_allow_html=True
        )

        def go_to_landing_page():
            st.session_state["show_landing_page"] = True
            st.session_state["logged_in"] = False
            st.query_params.clear()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.button(
            "← Back to Overview",
            key="login_back_home",
            use_container_width=True,
            on_click=go_to_landing_page
        )

    st.stop()


# ==================================================
# AUTHENTICATED APPLICATION SHELL & CALLBACKS
# ==================================================

def nav_to_troubleshoot():
    st.session_state["main_navigation_radio"] = "Troubleshoot"
    st.session_state["show_landing_page"] = False
    role_slug = (st.session_state.get("user_role") or "Technician").lower()
    st.query_params["view"] = "troubleshoot"
    st.query_params["role"] = role_slug


def nav_to_history(machine_filter: str | None = None):
    st.session_state["main_navigation_radio"] = "History"
    st.session_state["show_landing_page"] = False
    if machine_filter:
        st.session_state["history_filter_machine"] = machine_filter
    role_slug = (st.session_state.get("user_role") or "Technician").lower()
    st.query_params["view"] = "history"
    st.query_params["role"] = role_slug


def nav_to_documents():
    st.session_state["main_navigation_radio"] = "Documents"
    st.session_state["show_landing_page"] = False
    role_slug = (st.session_state.get("user_role") or "Technician").lower()
    st.query_params["view"] = "documents"
    st.query_params["role"] = role_slug


def do_logout():
    st.session_state["logged_in"] = False
    st.session_state["show_landing_page"] = False
    st.session_state["user_role"] = None
    st.session_state["username"] = None
    st.session_state["troubleshooting_result"] = None
    st.session_state["troubleshooting_input"] = {}
    st.session_state["history_detail"] = None
    st.session_state["show_feedback_comment"] = False
    st.session_state["main_navigation_radio"] = "Dashboard"
    st.query_params.clear()
    st.query_params["page"] = "login"


user_role = st.session_state.get("user_role", "Technician")
username = st.session_state.get("username", "technician")

# Sidebar Navigation
with st.sidebar:
    # 1. Branding Header
    st.markdown(
        clean_html("""
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 16px; margin-bottom: 22px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 1px 4px rgba(27,69,88,0.04);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #EF3837; color: #FFFFFF; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-family: 'Outfit', sans-serif;">
                    N
                </div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #1B4558; letter-spacing: -0.5px; line-height: 1.1;">NERI</div>
                    <div style="font-size: 0.675rem; font-weight: 700; color: #EF3837; margin-top: 1px;">Maintenance AI</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 4px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 20px; padding: 3px 8px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background-color: #10B981;"></span>
                <span style="font-size: 0.625rem; font-weight: 700; color: #065F46;">ONLINE</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    nav_options = ["Dashboard", "Troubleshoot", "History", "Documents"]

    view_param = (st.query_params.get("view") or "").lower()
    view_map = {
        "dashboard": "Dashboard",
        "troubleshoot": "Troubleshoot",
        "history": "History",
        "documents": "Documents"
    }

    if "main_navigation_radio" not in st.session_state or st.session_state["main_navigation_radio"] not in nav_options:
        if view_param in view_map:
            st.session_state["main_navigation_radio"] = view_map[view_param]
        else:
            st.session_state["main_navigation_radio"] = "Dashboard"
    elif view_param in view_map and st.session_state.get("_last_synced_view") != view_param:
        st.session_state["main_navigation_radio"] = view_map[view_param]
        st.session_state["_last_synced_view"] = view_param

    selected_nav = st.session_state["main_navigation_radio"]

    # Sidebar Navigation Styling and Active Highlighting
    sidebar_css_rules = """
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E5E7EB !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Zero-height marker to avoid spacing gaps */
        .sidebar-nav-marker,
        div.stElementContainer:has(.sidebar-nav-marker) {
            height: 0px !important;
            max-height: 0px !important;
            min-height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
            overflow: hidden !important;
            line-height: 0 !important;
            border: none !important;
        }

        /* Base styling for all sidebar nav buttons */
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker) + div.stElementContainer button,
        section[data-testid="stSidebar"] div.stButton > button {
            text-align: left !important;
            justify-content: flex-start !important;
            font-size: 0.9rem !important;
            padding: 10px 14px 10px 42px !important;
            border-radius: 8px !important;
            margin-bottom: 6px !important;
            position: relative !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
        }

        /* ACTIVE NAVIGATION OPTION (#F3E8FF background, 4px solid #EF3837 left indicator) */
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker[data-active="true"]) + div.stElementContainer button {
            background-color: #F3E8FF !important;
            background: #F3E8FF !important;
            color: #1B4558 !important;
            border-left: 4px solid #EF3837 !important;
            border-top: 1px solid #DDB7FF !important;
            border-right: 1px solid #DDB7FF !important;
            border-bottom: 1px solid #DDB7FF !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(221, 183, 255, 0.25) !important;
        }

        section[data-testid="stSidebar"] button[kind="primary"] *,
        section[data-testid="stSidebar"] button[kind="primary"] p,
        section[data-testid="stSidebar"] button[kind="primary"] span,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] *,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] p,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] span,
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker[data-active="true"]) + div.stElementContainer button * {
            color: #1B4558 !important;
            font-weight: 700 !important;
        }

        /* INACTIVE NAVIGATION OPTIONS (#FFFFFF background, muted text #5A7383) */
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker[data-active="false"]) + div.stElementContainer button,
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker) + div.stElementContainer button[kind="secondary"]:not([key="logout_btn"]) {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            color: #5A7383 !important;
            border: 1px solid #E5E7EB !important;
            font-weight: 500 !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker[data-active="false"]) + div.stElementContainer button *,
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker) + div.stElementContainer button[kind="secondary"]:not([key="logout_btn"]) * {
            color: #5A7383 !important;
            font-weight: 500 !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker[data-active="false"]) + div.stElementContainer button:hover,
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker) + div.stElementContainer button[kind="secondary"]:not([key="logout_btn"]):hover {
            background-color: #F8F3FC !important;
            color: #1B4558 !important;
            border-color: #DDB7FF !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker[data-active="false"]) + div.stElementContainer button:hover *,
        section[data-testid="stSidebar"] div.stElementContainer:has(.sidebar-nav-marker) + div.stElementContainer button[kind="secondary"]:not([key="logout_btn"]):hover * {
            color: #1B4558 !important;
        }

        /* PER-ITEM SVG ICONS */
        /* Dashboard */
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Dashboard"]) + div.stElementContainer button::before {
            content: "" !important;
            position: absolute !important;
            left: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 18px !important;
            height: 18px !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235A7383' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='9'/%3E%3Crect x='14' y='3' width='7' height='5'/%3E%3Crect x='14' y='12' width='7' height='9'/%3E%3Crect x='3' y='15' width='7' height='6'/%3E%3C/svg%3E") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
        }
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Dashboard"][data-active="true"]) + div.stElementContainer button::before,
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Dashboard"]) + div.stElementContainer button[kind="primary"]::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23EF3837' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='9'/%3E%3Crect x='14' y='3' width='7' height='5'/%3E%3Crect x='14' y='12' width='7' height='9'/%3E%3Crect x='3' y='15' width='7' height='6'/%3E%3C/svg%3E") !important;
        }

        /* Troubleshoot */
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Troubleshoot"]) + div.stElementContainer button::before {
            content: "" !important;
            position: absolute !important;
            left: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 18px !important;
            height: 18px !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235A7383' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z'/%3E%3C/svg%3E") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
        }
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Troubleshoot"][data-active="true"]) + div.stElementContainer button::before,
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Troubleshoot"]) + div.stElementContainer button[kind="primary"]::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23EF3837' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z'/%3E%3C/svg%3E") !important;
        }

        /* History */
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="History"]) + div.stElementContainer button::before {
            content: "" !important;
            position: absolute !important;
            left: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 18px !important;
            height: 18px !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235A7383' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
        }
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="History"][data-active="true"]) + div.stElementContainer button::before,
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="History"]) + div.stElementContainer button[kind="primary"]::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23EF3837' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E") !important;
        }

        /* Documents */
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Documents"]) + div.stElementContainer button::before {
            content: "" !important;
            position: absolute !important;
            left: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 18px !important;
            height: 18px !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235A7383' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3C/svg%3E") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
        }
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Documents"][data-active="true"]) + div.stElementContainer button::before,
        div.stElementContainer:has(.sidebar-nav-marker[data-nav="Documents"]) + div.stElementContainer button[kind="primary"]::before {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23EF3837' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3C/svg%3E") !important;
        }

        /* LOGOUT BUTTON (#FEF2F2 surface, #EF3837 text & icon) */
        .logout-marker,
        div.stElementContainer:has(.logout-marker) {
            height: 0px !important;
            max-height: 0px !important;
            min-height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
            overflow: hidden !important;
            line-height: 0 !important;
            border: none !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.logout-marker) + div.stElementContainer button,
        section[data-testid="stSidebar"] button[key="logout_btn"] {
            background-color: #FEF2F2 !important;
            background: #FEF2F2 !important;
            color: #EF3837 !important;
            border: 1px solid #FCA5A5 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 10px 14px 10px 42px !important;
            margin-top: 14px !important;
            position: relative !important;
            text-align: left !important;
            justify-content: flex-start !important;
            transition: all 0.2s ease !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.logout-marker) + div.stElementContainer button::before {
            content: "" !important;
            position: absolute !important;
            left: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 16px !important;
            height: 16px !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23EF3837' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4'/%3E%3Cpolyline points='16 17 21 12 16 7'/%3E%3Cline x1='21' y1='12' x2='9' y2='12'/%3E%3C/svg%3E") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.logout-marker) + div.stElementContainer button * {
            color: #EF3837 !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.logout-marker) + div.stElementContainer button:hover {
            background-color: #FEE2E2 !important;
            color: #D92D2C !important;
            border-color: #EF3837 !important;
        }

        section[data-testid="stSidebar"] div.stElementContainer:has(.logout-marker) + div.stElementContainer button:hover * {
            color: #D92D2C !important;
        }
    """

    st.markdown(clean_html(f"<style>{sidebar_css_rules}</style>"), unsafe_allow_html=True)

    for item in nav_options:
        is_active = (item == selected_nav)
        btn_key = f"nav_btn_{item}"
        btn_type: Literal["primary", "secondary"] = "primary" if is_active else "secondary"

        st.markdown(
            f'<div class="sidebar-nav-marker" data-nav="{item}" data-active="{str(is_active).lower()}"></div>',
            unsafe_allow_html=True
        )
        if st.button(item, key=btn_key, use_container_width=True, type=btn_type):
            if st.session_state.get("main_navigation_radio") != item:
                st.session_state["main_navigation_radio"] = item
                view_slug = item.lower()
                role_slug = (st.session_state.get("user_role") or "Technician").lower()
                st.query_params["view"] = view_slug
                st.query_params["role"] = role_slug
                st.session_state["_last_synced_view"] = view_slug
                st.rerun()

    if st.session_state.get("logged_in") and selected_nav:
        view_slug = selected_nav.lower()
        role_slug = (st.session_state.get("user_role") or "Technician").lower()
        if st.query_params.get("view") != view_slug or st.query_params.get("role") != role_slug:
            st.query_params["view"] = view_slug
            st.query_params["role"] = role_slug

    # User Profile Card at Bottom of Sidebar
    role_pill_bg = "#FEE2E2" if user_role == "Supervisor" else "#F3E8FF"
    role_pill_color = "#EF3837" if user_role == "Supervisor" else "#1B4558"
    role_pill_border = "#FCA5A5" if user_role == "Supervisor" else "#DDB7FF"

    st.markdown(
        clean_html(f"""
        <div style="margin-top: 36px; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px; box-shadow: 0 1px 4px rgba(27,69,88,0.04);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: #F8F3FC; border: 1px solid #E5E7EB; padding: 7px; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    {get_svg_icon("user", color="#1B4558", size=18)}
                </div>
                <div>
                    <div style="font-size: 0.875rem; font-weight: 700; color: #1B4558; line-height: 1.2;">{username}</div>
                    <div style="margin-top: 4px;">
                        <span style="font-size: 0.675rem; font-weight: 700; color: {role_pill_color}; background: {role_pill_bg}; border: 1px solid {role_pill_border}; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">
                            {user_role}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown('<div class="logout-marker"></div>', unsafe_allow_html=True)
    st.button("Sign Out", use_container_width=True, type="secondary", key="logout_btn", on_click=do_logout)

# Top Header Bar in Main Layout
st.markdown(
    clean_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 14px; border-bottom: 1px solid #E5E7EB; margin-bottom: 22px;">
        <div>
            <span style="font-size: 0.725rem; font-weight: 700; color: #5A7383; text-transform: uppercase; letter-spacing: 0.5px;">NERI WORKSPACE</span>
            <h2 style="font-size: 1.55rem; font-weight: 800; color: #1B4558; margin: 0;">{selected_nav}</h2>
        </div>
        <div class="status-pill active">
            {get_svg_icon("server", color="#6B21A8", size=14)} Knowledge Base Online
        </div>
    </div>
    """),
    unsafe_allow_html=True
)

# ==================================================
# PAGE 1: DASHBOARD
# ==================================================

if selected_nav == "Dashboard":

    st.markdown(
        clean_html(f"""
        <div class="neri-card" style="border-left: 4px solid #EF3837;">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: #1B4558; margin-bottom: 4px;">Welcome, {username}</h3>
            <p style="font-size: 0.875rem; color: #5A7383; margin: 0;">
                Industrial maintenance troubleshooting assistant for plant machinery, manuals, and safety records.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    total_sessions_count = 0
    approved_docs_count = 0
    unique_machines = set()

    try:
        sessions = fetch_history_sessions()
        total_sessions_count = len(sessions)
        for s in sessions:
            if s.get("machine"):
                unique_machines.add(s.get("machine"))
    except Exception:
        pass

    try:
        docs = fetch_documents_list()
        approved_docs_count = len(docs)
    except Exception:
        pass

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Active Machines</span>
                    {get_svg_icon("wrench", color="#EF3837", size=20)}
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #1B4558;">{len(unique_machines) if unique_machines else '—'}</div>
                <div style="font-size: 0.75rem; color: #5A7383; margin-top: 4px;">Indexed plant machinery</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Diagnostic Queries</span>
                    {get_svg_icon("history", color="#EF3837", size=20)}
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #1B4558;">{total_sessions_count}</div>
                <div style="font-size: 0.75rem; color: #5A7383; margin-top: 4px;">Recorded troubleshooting cases</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m_col3:
        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Approved Documents</span>
                    {get_svg_icon("document", color="#EF3837", size=20)}
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #1B4558;">{approved_docs_count}</div>
                <div style="font-size: 0.75rem; color: #5A7383; margin-top: 4px;">Indexed manuals, logs & specs</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Operational Command Action Panel
    st.markdown(
        clean_html("""
        <div class="neri-card" style="border-left: 4px solid #EF3837;">
            <div style="font-size: 0.725rem; font-weight: 700; color: #EF3837; text-transform: uppercase; margin-bottom: 4px;">OPERATIONAL COMMAND</div>
            <h4 style="font-size: 1.15rem; font-weight: 800; color: #1B4558; margin: 0 0 6px 0;">Diagnose a Machine Malfunction</h4>
            <p style="font-size: 0.875rem; color: #5A7383; margin: 0 0 14px 0; line-height: 1.5;">
                Start a diagnostic session grounded in approved OEM manuals, maintenance history, and plant safety specs.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.button("Start Troubleshooting Session →", type="primary", key="dash_start_tb", on_click=nav_to_troubleshoot)

    # Recent Troubleshooting Activity
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        clean_html(f"""
        <div class="neri-card-header">
            {get_svg_icon("clock", color="#1B4558", size=20)} Recent Diagnostic Cases
        </div>
        """),
        unsafe_allow_html=True
    )

    try:
        recent_sessions = fetch_history_sessions()[:5]
        if not recent_sessions:
            st.info("No recent troubleshooting sessions recorded yet.")
        else:
            for s in recent_sessions:
                m_name = s.get("machine", "Unknown Machine")
                m_id = s.get("machine_id", "N/A")
                prob = s.get("problem", "N/A")
                dt = s.get("created_at", "")[:16].replace("T", " ")
                fb = s.get("feedback")
                fb_badge = f'<span class="status-pill active">{fb.title()}</span>' if fb else '<span style="color:#8CA0AD; font-size:0.75rem;">No Feedback</span>'

                st.markdown(
                    clean_html(f"""
                    <div class="document-box">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1B4558;">{m_name}</strong> 
                                <span style="color: #5A7383; font-size: 0.8rem;">(ID: {m_id})</span>
                                <div style="font-size: 0.85rem; color: #5A7383; margin-top: 2px;">{prob[:90]}{'...' if len(prob) > 90 else ''}</div>
                            </div>
                            <div style="text-align: right; flex-shrink: 0;">
                                <div style="font-size: 0.75rem; color: #5A7383; margin-bottom: 4px;">{dt}</div>
                                {fb_badge}
                            </div>
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
    except Exception:
        st.error("Could not fetch recent activity.")


# ==================================================
# PAGE 2: TROUBLESHOOTING
# ==================================================

elif selected_nav == "Troubleshoot":

    st.markdown(
        clean_html("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 0.725rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px;">DIAGNOSTIC WORKSPACE</div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #1B4558; margin: 2px 0 4px 0;">Analyze Machine Malfunction</h3>
            <p style="font-size: 0.875rem; color: #5A7383; margin: 0;">
                Describe observed symptoms and Neri will search approved documentation for grounded guidance and safety rules.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    tb_left, tb_right = st.columns([1.4, 0.8], gap="large")

    with tb_left:
        with st.form("troubleshoot_form"):
            f_col1, f_col2 = st.columns(2)

            with f_col1:
                machine = st.text_input("Machine", placeholder="e.g. Hydraulic Press or CNC Mill")

            with f_col2:
                machine_id = st.text_input("Machine ID", placeholder="e.g. PRS-001 or CNC-204")

            problem = st.text_area(
                "Problem Description & Observed Symptoms",
                placeholder="Describe observed machine symptoms, pressure drops, leaks, unusual noise, vibrations...",
                height=120
            )

            error_code = st.text_input("Error Code (optional)", placeholder="e.g. E-115 or ALM-04")

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            submit_troubleshoot = st.form_submit_button("Analyze Problem →", type="primary", use_container_width=True)

            if submit_troubleshoot:
                if not machine.strip() or not machine_id.strip() or not problem.strip():
                    st.error("Please provide the Machine, Machine ID, and Problem description.")
                else:
                    payload = {
                        "machine": machine.strip(),
                        "machine_id": machine_id.strip(),
                        "problem": problem.strip(),
                        "error_code": error_code.strip() if error_code.strip() else None,
                    }

                    progress_placeholder = st.empty()
                    progress_placeholder.markdown(
                        clean_html(f"""
                        <div class="neri-card" style="border-left: 4px solid #EF3837; padding: 20px;">
                            <div style="font-weight: 700; font-size: 1rem; color: #1B4558; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">
                                {get_svg_icon("cpu", color="#EF3837", size=20)} 
                                <span>Running Grounded Diagnostic Pipeline...</span>
                            </div>
                            <div style="font-size: 0.85rem; color: #5A7383; line-height: 1.8; display: flex; flex-direction: column; gap: 6px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("search", color="#EF3837", size=14)} <span>Normalizing machine ID & searching ChromaDB...</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("shield", color="#EF3837", size=14)} <span>Evaluating safety hazard guardrails...</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("check", color="#10B981", size=14)} <strong style="color: #1B4558;">Synthesizing citation-backed steps...</strong>
                                </div>
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True
                    )

                    try:
                        res_json = post_troubleshoot_query(payload)
                        progress_placeholder.empty()

                        if not res_json:
                            st.error("Neri could not process the troubleshooting request.")
                        else:
                            st.session_state["troubleshooting_result"] = res_json
                            st.session_state["troubleshooting_input"] = payload
                            st.session_state["show_feedback_comment"] = False
                            st.rerun()

                    except Exception as error:
                        progress_placeholder.empty()
                        st.error(f"Could not process troubleshooting request: {error}")

    with tb_right:
        st.markdown(
            clean_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; padding: 22px; box-shadow: 0 1px 4px rgba(27,69,88,0.03);">
                <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.95rem; color: #1B4558; margin-bottom: 12px;">
                    {get_svg_icon("shield-check", color="#EF3837", size=18)} Safe Troubleshooting Protocols
                </div>
                <div style="font-size: 0.825rem; color: #5A7383; line-height: 1.5; display: flex; flex-direction: column; gap: 10px;">
                    <div><strong>1. Machine Identification:</strong> Include exact machine name and plant ID for precise document retrieval.</div>
                    <div><strong>2. Symptom Details:</strong> Detail pressure readings, leaks, strange vibrations, or visual indicators.</div>
                    <div><strong>3. Error Codes:</strong> Enter control panel alarm codes if displayed (e.g. E-115).</div>
                    <div style="background: #FFFBEB; padding: 12px; border-radius: 8px; border: 1px solid #FCD34D; color: #92400E; margin-top: 4px;">
                        <strong>Plant Safety Mandate:</strong> Strictly adhere to Lockout/Tagout (LOTO) protocols prior to touching high-voltage or pressurized lines.
                    </div>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    # Display Diagnostic Result Report
    if st.session_state.get("troubleshooting_result") is not None:
        result = st.session_state["troubleshooting_result"]
        input_data = st.session_state.get("troubleshooting_input", {})

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.divider()

        is_reliable = result.get("reliable", False)

        if not is_reliable:
            safety_warning = result.get("safety_warning")
            message_text = result.get("message", "We couldn't find enough information in approved documentation to provide a reliable answer.")
            
            if safety_warning:
                # SAFETY-CRITICAL REFUSAL STATE
                st.markdown(
                    clean_html(f"""
                    <div class="critical-box">
                        <div class="critical-box-title">
                            {get_svg_icon("alert", color="#EF3837", size=22)} SAFETY DOCUMENTATION REQUIRED &bull; CRITICAL HAZARD REFUSAL
                        </div>
                        <div style="font-size: 0.925rem; font-weight: 600; line-height: 1.6; margin-bottom: 12px; color: #991B1B;">
                            {message_text}
                        </div>
                        <div style="font-size: 0.825rem; color: #991B1B; background: #FEE2E2; padding: 12px; border-radius: 8px; border: 1px solid #FCA5A5;">
                            <strong>Safety Enforcement:</strong> Neri strictly halts troubleshooting for hazard-classified queries when certified safety procedures are missing or incomplete (LOTO Protocol Enforcement).
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
            else:
                # NO RELIABLE ANSWER STATE
                st.markdown(
                    clean_html(f"""
                    <div class="neri-card" style="border-left: 4px solid #EF3837; background: #FFFFFF; padding: 22px;">
                        <div style="display: flex; align-items: center; gap: 10px; font-size: 1.1rem; font-weight: 700; color: #EF3837; margin-bottom: 8px;">
                            {get_svg_icon("document", color="#EF3837", size=22)} No Reliable Documentation Found
                        </div>
                        <div style="font-size: 0.9rem; color: #5A7383; line-height: 1.6; margin-bottom: 12px;">
                            {message_text}
                        </div>
                        <div style="font-size: 0.825rem; color: #991B1B; background: #FEF2F2; border: 1px solid #FCA5A5; padding: 12px; border-radius: 8px; font-weight: 500;">
                            Try refining machine symptoms or confirm whether the required maintenance manual is uploaded to the Knowledge Base.
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )

        else:
            # DIAGNOSTIC REPORT STATE
            m_name = result.get("machine") or input_data.get("machine", "Equipment")
            m_id = result.get("machine_id") or input_data.get("machine_id", "N/A")
            prob_txt = result.get("problem") or input_data.get("problem", "N/A")
            err_code = result.get("error_code") or input_data.get("error_code")

            err_code_badge = f'<span style="background: #F1F5F9; color: #5A7383; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">Code: {err_code}</span>' if err_code else ''

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-bottom: 16px; padding: 22px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;">
                        <div>
                            <div style="font-size: 0.725rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px;">DIAGNOSTIC GUIDANCE REPORT</div>
                            <h3 style="font-size: 1.35rem; font-weight: 800; color: #1B4558; margin: 4px 0 8px 0;">Troubleshooting Recommendations</h3>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #1B4558; display: flex; align-items: center; gap: 8px;">
                                <span>{m_name}</span>
                                <span style="color: #CBD5E1;">•</span>
                                <span style="color: #5A7383;">ID: {m_id}</span>
                                {err_code_badge}
                            </div>
                            <div style="font-size: 0.875rem; color: #5A7383; margin-top: 6px; line-height: 1.4;">
                                <strong style="color: #1B4558;">Problem:</strong> {prob_txt}
                            </div>
                        </div>
                        <span class="status-pill active" style="flex-shrink: 0;">
                            {get_svg_icon("shield-check", color="#6B21A8", size=14)} Grounded in Documentation
                        </span>
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

            # Prominent Safety Warning Box
            safety = result.get("safety_warning")
            if safety:
                safety_msg = safety.get("message", "") if isinstance(safety, dict) else str(safety)
                if safety_msg.strip():
                    st.markdown(
                        clean_html(f"""
                        <div class="warning-box" style="margin-bottom: 16px;">
                            <div class="warning-box-title">
                                {get_svg_icon("alert", color="#92400E", size=20)} MANDATORY SAFETY WARNING & PROTOCOLS
                            </div>
                            <div style="font-size: 0.9rem; font-weight: 600; line-height: 1.5; color: #92400E;">
                                {safety_msg}
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True
                    )

            # Possible Causes
            causes = result.get("possible_causes", [])
            causes_html = []
            if causes:
                for idx, cause in enumerate(causes, start=1):
                    cause_str = cause.get("cause", "") if isinstance(cause, dict) else str(cause)
                    causes_html.append(f"""
                    <div class="neri-step-item">
                        <div class="neri-badge-num">0{idx}</div>
                        <div style="font-size: 0.9rem; font-weight: 600; color: #1B4558; margin-top: 4px;">{cause_str}</div>
                    </div>
                    """)
                causes_body = "".join(causes_html)
            else:
                causes_body = '<div style="font-size: 0.875rem; color: #5A7383; padding: 12px 0;">No documented causes found.</div>'

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-bottom: 16px;">
                    <div class="neri-card-header" style="border-bottom: 1px solid #F3F4F6; padding-bottom: 12px; margin-bottom: 8px;">
                        {get_svg_icon("search", color="#1B4558", size=18)} Documented Possible Causes
                    </div>
                    {causes_body}
                </div>
                """),
                unsafe_allow_html=True
            )

            # Troubleshooting Steps Timeline
            steps = result.get("troubleshooting_steps", [])
            steps_html = []
            if steps:
                for idx, step in enumerate(steps, start=1):
                    step_str = step.get("step", "") if isinstance(step, dict) else str(step)
                    steps_html.append(f"""
                    <div class="neri-step-item">
                        <div class="neri-badge-num">0{idx}</div>
                        <div style="font-size: 0.9rem; color: #1B4558; line-height: 1.5; margin-top: 4px;">{step_str}</div>
                    </div>
                    """)
                steps_body = "".join(steps_html)
            else:
                steps_body = '<div style="font-size: 0.875rem; color: #5A7383; padding: 12px 0;">No documented troubleshooting steps found.</div>'

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-bottom: 16px;">
                    <div class="neri-card-header" style="border-bottom: 1px solid #F3F4F6; padding-bottom: 12px; margin-bottom: 8px;">
                        {get_svg_icon("wrench", color="#1B4558", size=18)} Recommended Troubleshooting Steps
                    </div>
                    {steps_body}
                </div>
                """),
                unsafe_allow_html=True
            )

            # Source References (Grouped by Document)
            sources = result.get("sources", [])
            if sources:
                grouped_sources: dict[str, dict] = {}
                for src in sources:
                    doc_name = src.get("document") or "Unknown document"
                    if doc_name not in grouped_sources:
                        raw_type = src.get("document_type")
                        if not raw_type or str(raw_type).lower() in ["unknown", "n/a", "none", "null"]:
                            formatted_type = "MANUAL"
                        else:
                            formatted_type = str(raw_type).replace("_", " ").upper()

                        grouped_sources[doc_name] = {
                            "document": doc_name,
                            "document_type": formatted_type,
                            "sections": set(),
                            "pages": set(),
                            "count": 0
                        }
                    
                    grouped_sources[doc_name]["count"] += 1
                    sec = src.get("section")
                    if sec and str(sec).lower() not in ["unknown", "n/a", "none", "null"]:
                        grouped_sources[doc_name]["sections"].add(str(sec))
                    pg = src.get("page")
                    if pg and str(pg).lower() not in ["unknown", "n/a", "none", "null"]:
                        grouped_sources[doc_name]["pages"].add(str(pg))

                src_cards_html = []
                for doc_name, info in grouped_sources.items():
                    doc_type_str = info["document_type"]
                    count_str = f"{info['count']} retrieved section" + ("s" if info['count'] > 1 else "")
                    
                    sec_list = sorted(info["sections"])
                    sec_val = f"Section: {', '.join(sec_list)}" if sec_list else "Section: Specified in manual"
                    
                    pg_list = sorted(info["pages"])
                    pg_val = f"Page: {', '.join(pg_list)}" if pg_list else "Page: Specified in manual"

                    src_cards_html.append(f"""
                    <div class="source-box" style="margin-bottom: 10px;">
                        <div style="font-weight: 700; font-size: 0.9rem; color: #1B4558; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                {get_svg_icon("file-text", color="#EF3837", size=18)}
                                <span>{doc_name}</span>
                            </div>
                            <span style="font-size: 0.725rem; font-weight: 700; color: #6B21A8; background: #F3E8FF; padding: 2px 8px; border-radius: 4px;">{doc_type_str}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #5A7383; display: flex; gap: 16px;">
                            <span>{sec_val}</span>
                            <span>{pg_val}</span>
                            <span style="color: #EF3837; font-weight: 600;">{count_str}</span>
                        </div>
                    </div>
                    """)
                
                sources_body = "".join(src_cards_html)
            else:
                sources_body = '<div style="font-size: 0.875rem; color: #5A7383; padding: 12px 0;">No source references available.</div>'

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-top: 16px; margin-bottom: 16px; padding: 22px;">
                    <div class="neri-card-header" style="border-bottom: 1px solid #F3F4F6; padding-bottom: 12px; margin-bottom: 12px;">
                        {get_svg_icon("document", color="#1B4558", size=18)} Verified Source References
                    </div>
                    {sources_body}
                </div>
                """),
                unsafe_allow_html=True
            )

            # Feedback Section
            st.markdown(
                clean_html("""
                <div class="neri-card" style="margin-bottom: 12px; padding: 20px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1B4558; margin-bottom: 12px;">
                        Was this troubleshooting guidance helpful?
                    </div>
                """),
                unsafe_allow_html=True
            )

            session_id = result.get("session_id")
            fb_col1, fb_col2 = st.columns(2)

            with fb_col1:
                if st.button("Helpful Guidance", use_container_width=True, key="fb_helpful_btn"):
                    if session_id:
                        try:
                            if post_feedback(session_id, "helpful"):
                                st.success("Thank you! Feedback recorded.")
                        except Exception:
                            st.error("Could not record feedback.")

            with fb_col2:
                if st.button("Not Helpful", use_container_width=True, key="fb_not_helpful_btn"):
                    st.session_state["show_feedback_comment"] = True
                    st.rerun()

            if st.session_state.get("show_feedback_comment", False):
                comment_text = st.text_area("What could be improved?", placeholder="Tell us what was missing or inaccurate...", key="fb_comment_area")
                if st.button("Submit Feedback", type="primary", key="fb_submit_btn"):
                    if session_id:
                        try:
                            if post_feedback(session_id, "not_helpful", comment_text.strip() if comment_text.strip() else None):
                                st.success("Feedback submitted.")
                                st.session_state["show_feedback_comment"] = False
                                st.rerun()
                        except Exception:
                            st.error("Could not submit feedback.")

            st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# PAGE 3: HISTORY
# ==================================================

elif selected_nav == "History":

    st.markdown(
        clean_html("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 0.725rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px;">CASE ARCHIVE</div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #1B4558; margin: 2px 0 4px 0;">Troubleshooting Case History</h3>
            <p style="font-size: 0.875rem; color: #5A7383; margin: 0;">
                Review archived diagnostic sessions, machine symptoms, and plant technical guidance.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    try:
        sessions = fetch_history_sessions()
        total_sessions = len(sessions)
        helpful_sessions = len([s for s in sessions if s.get("feedback") == "helpful"])
        unique_machines_count = len({s.get("machine") for s in sessions if s.get("machine")})

        hk_col1, hk_col2, hk_col3 = st.columns(3)
        with hk_col1:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Total Cases</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1B4558; margin-top: 4px;">{total_sessions}</div>
            </div>
            """), unsafe_allow_html=True)
        with hk_col2:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Helpful Feedback</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #EF3837; margin-top: 4px;">{helpful_sessions}</div>
            </div>
            """), unsafe_allow_html=True)
        with hk_col3:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Unique Equipment</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1B4558; margin-top: 4px;">{unique_machines_count}</div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        if not sessions:
            st.info("No troubleshooting sessions found in history.")
        else:
            if st.session_state.get("history_filter_machine"):
                st.caption(f"Filtering history by machine: {st.session_state['history_filter_machine']}")
                if st.button("Clear Machine Filter", key="clear_hist_filter"):
                    st.session_state["history_filter_machine"] = None
                    st.rerun()

            for session in sessions:
                s_id = session.get("id")
                m_name = session.get("machine", "Unknown Machine")
                m_id = session.get("machine_id", "N/A")
                prob = session.get("problem", "N/A")
                dt = session.get("created_at", "")[:16].replace("T", " ")
                fb = session.get("feedback")
                fb_str = f"Feedback: {fb.title()}" if fb else "Feedback: None"

                exp_label = f"{m_name} (ID: {m_id}) — {prob[:60]}... [{dt}]"

                with st.expander(exp_label):
                    st.markdown(
                        clean_html(f"""
                        <div style="font-size: 0.875rem; color: #1B4558; line-height: 1.6; margin-bottom: 12px;">
                            <strong>Date/Time:</strong> {dt}<br>
                            <strong>Machine:</strong> {m_name} ({m_id})<br>
                            <strong>Problem:</strong> {prob}<br>
                            {"<strong>Error Code:</strong> " + session['error_code'] + "<br>" if session.get('error_code') else ""}
                            <strong>Status:</strong> {fb_str}
                        </div>
                        """),
                        unsafe_allow_html=True
                    )

                    if s_id is not None and st.button("View Archived Diagnostic Report", key=f"hist_view_{s_id}"):
                        try:
                            det = fetch_session_detail(int(s_id))
                            if det:
                                st.session_state["history_detail"] = det
                                st.rerun()
                        except Exception:
                            st.error("Could not load session details.")

    except Exception:
        st.error("Could not load troubleshooting history.")

    # Detail View Report Modal Section
    if st.session_state.get("history_detail") is not None:
        det = st.session_state["history_detail"]
        resp_data = det.get("response", {})

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.divider()

        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="border-left: 4px solid #EF3837;">
                <div class="neri-card-header">
                    {get_svg_icon("document", color="#EF3837", size=20)} Archived Diagnostic Case Report
                </div>
                <div style="font-size: 0.875rem; color: #1B4558; line-height: 1.6;">
                    <strong>Machine:</strong> {det.get('machine')}<br>
                    <strong>Machine ID:</strong> {det.get('machine_id')}<br>
                    <strong>Problem:</strong> {det.get('problem')}<br>
                    {"<strong>Error Code:</strong> " + det['error_code'] + "<br>" if det.get('error_code') else ""}
                    <strong>Recorded At:</strong> {det.get('created_at', '')[:19].replace('T', ' ')}
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

        if not resp_data.get("reliable", False):
            st.warning(resp_data.get("message", "No reliable answer was found."))
        else:
            # Causes
            causes_list = resp_data.get("possible_causes", [])
            causes_markup = []
            for idx, c in enumerate(causes_list, start=1):
                c_str = c.get("cause", "") if isinstance(c, dict) else str(c)
                causes_markup.append(f"""
                <div class="neri-step-item">
                    <div class="neri-badge-num">0{idx}</div>
                    <div style="font-size: 0.875rem; font-weight: 600; color: #1B4558; margin-top: 4px;">{c_str}</div>
                </div>
                """)

            st.markdown(
                clean_html(f"""
                <div class="neri-card">
                    <div class="neri-card-header">{get_svg_icon("search", color="#1B4558", size=18)} Archived Causes</div>
                    {''.join(causes_markup)}
                </div>
                """),
                unsafe_allow_html=True
            )

            # Steps
            steps_list = resp_data.get("troubleshooting_steps", [])
            steps_markup = []
            for idx, s in enumerate(steps_list, start=1):
                s_str = s.get("step", "") if isinstance(s, dict) else str(s)
                steps_markup.append(f"""
                <div class="neri-step-item">
                    <div class="neri-badge-num">0{idx}</div>
                    <div style="font-size: 0.875rem; color: #1B4558; line-height: 1.5; margin-top: 4px;">{s_str}</div>
                </div>
                """)

            st.markdown(
                clean_html(f"""
                <div class="neri-card">
                    <div class="neri-card-header">{get_svg_icon("wrench", color="#1B4558", size=18)} Archived Steps</div>
                    {''.join(steps_markup)}
                </div>
                """),
                unsafe_allow_html=True
            )

        if st.button("Close Case Report", key="close_case_report"):
            st.session_state["history_detail"] = None
            st.rerun()


# ==================================================
# PAGE 4: DOCUMENTS
# ==================================================

elif selected_nav == "Documents":

    st.markdown(
        clean_html("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 0.725rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px;">KNOWLEDGE BASE</div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #1B4558; margin: 2px 0 4px 0;">Approved Documentation Management</h3>
            <p style="font-size: 0.875rem; color: #5A7383; margin: 0;">
                Review approved manuals, maintenance logs, and safety procedures indexed in the Neri Knowledge Base.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    try:
        all_docs = fetch_documents_list()
        total_docs = len(all_docs)
        manuals_count = len([d for d in all_docs if "manual" in str(d.get("document_type", "")).lower()])
        logs_count = len([d for d in all_docs if "log" in str(d.get("document_type", "")).lower()])
        safety_count = len([d for d in all_docs if "safety" in str(d.get("document_type", "")).lower()])

        dk_col1, dk_col2, dk_col3, dk_col4 = st.columns(4)
        with dk_col1:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Total Docs</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1B4558; margin-top: 4px;">{total_docs}</div>
            </div>
            """), unsafe_allow_html=True)
        with dk_col2:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Manuals</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #EF3837; margin-top: 4px;">{manuals_count}</div>
            </div>
            """), unsafe_allow_html=True)
        with dk_col3:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Logs</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1B4558; margin-top: 4px;">{logs_count}</div>
            </div>
            """), unsafe_allow_html=True)
        with dk_col4:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #5A7383; text-transform: uppercase;">Safety Specs</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #D97706; margin-top: 4px;">{safety_count}</div>
            </div>
            """), unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Supervisor Role: Add Approved Documentation Form
    if user_role == "Supervisor":
        with st.expander("Add Approved Documentation", expanded=False):
            with st.form("doc_upload_form"):
                st.markdown("<h4 style='color: #1B4558; margin: 0 0 12px 0;'>Upload Document & Index into ChromaDB</h4>", unsafe_allow_html=True)

                doc_type = st.selectbox("Document Type", ["Manual", "Maintenance Log", "Safety Procedure"])

                uc1, uc2 = st.columns(2)
                with uc1:
                    machine_tag = st.text_input("Machine (Optional)", placeholder="e.g. CNC Mill")
                with uc2:
                    version_tag = st.text_input("Version (Optional)", placeholder="e.g. 1.0")

                owner_tag = st.text_input("Owner (Optional)", placeholder="e.g. Maintenance Dept")

                up_file = st.file_uploader("Select PDF or TXT file", type=["pdf", "txt"])

                submit_upload = st.form_submit_button("Upload & Index Document →", type="primary", use_container_width=True)

                if submit_upload:
                    if up_file is None:
                        st.error("Please select a valid PDF or TXT file to upload.")
                    else:
                        type_map = {
                            "Manual": "manual",
                            "Maintenance Log": "maintenance_log",
                            "Safety Procedure": "safety"
                        }
                        try:
                            with st.spinner("Indexing document into Knowledge Base..."):
                                res_json = post_document_upload(
                                    file_name=up_file.name,
                                    file_bytes=up_file.getvalue(),
                                    document_type=type_map[doc_type],
                                    machine=machine_tag.strip() if machine_tag.strip() else None,
                                    version=version_tag.strip() if version_tag.strip() else None,
                                    owner=owner_tag.strip() if owner_tag.strip() else None,
                                )

                            if res_json:
                                st.success("Document successfully uploaded and indexed!")
                                st.markdown(
                                    clean_html(f"""
                                    <div class="success-box">
                                        <strong>Document:</strong> {res_json.get('document', up_file.name)}<br>
                                        <strong>Chunks Created:</strong> {res_json.get('chunks', 0)}<br>
                                        <strong>Status:</strong> {res_json.get('status', 'indexed')}
                                    </div>
                                    """),
                                    unsafe_allow_html=True
                                )
                                st.rerun()
                            else:
                                st.error("Upload failed.")

                        except Exception as e:
                            st.error(f"Error indexing document: {e}")

    # Search / Filter Bar (Horizontal Filter Boxes)
    f_col1, f_col2 = st.columns([1, 1.45])
    with f_col1:
        search_query = st.text_input("Search documents", placeholder="Filter by document name or machine...", key="doc_search_input")
    with f_col2:
        raw_filter = st.segmented_control(
            "Filter Type",
            options=["All Types", "Manual", "Maintenance Log", "Safety Procedure"],
            default="All Types",
            key="doc_type_filter",
            label_visibility="visible",
        )
        type_filter = str(raw_filter) if raw_filter else "All Types"

    try:
        docs = fetch_documents_list()
    except Exception:
        docs = []
        st.error("Could not fetch document list.")

    if docs:
        if search_query.strip():
            q = search_query.strip().lower()
            words = [w for w in q.split() if len(w) > 1]

            def matches_query(d):
                fn = (d.get("filename", "") or "").lower()
                mc = (d.get("machine", "") or "").lower()
                dt = (d.get("document_type", "") or "").lower()
                combined = f"{fn} {mc} {dt}"
                if q in combined:
                    return True
                for w in words:
                    stem = w[:4] if len(w) >= 4 else w
                    if stem in combined:
                        return True
                    return False

            docs = [d for d in docs if matches_query(d)]

        if type_filter != "All Types":
            filter_type_map = {
                "Manual": "manual",
                "Maintenance Log": "maintenance_log",
                "Safety Procedure": "safety",
            }
            target_type = filter_type_map.get(type_filter, type_filter.lower().replace(" ", "_"))
            docs = [
                d for d in docs
                if target_type in str(d.get("document_type") or "").lower()
                or str(d.get("document_type") or "").lower() in target_type
            ]

        if not docs:
            st.info("No matching documents found in knowledge base.")
        else:
            for d in docs:
                try:
                    fname = d.get("filename", "Unknown")
                    st_val = str(d.get("status", "indexed")).lower()
                    if st_val == "indexed":
                        st_pill = '<span class="status-pill active">Indexed</span>'
                    elif st_val == "processing":
                        st_pill = '<span class="status-pill warning">Processing</span>'
                    else:
                        st_pill = '<span class="status-pill error">Error</span>'

                    doc_type = str(d.get("document_type", "manual")).replace("_", " ").title()
                    machine = d.get("machine") or "All Equipment"
                    version = d.get("version") or "1.0"
                    chunks = d.get("chunks", 0)
                    target_doc_id = d.get("doc_id") or fname

                    exp_label = f"{fname}  •  {doc_type}  •  {machine}  ({chunks} Chunks)"

                    with st.expander(exp_label):
                        doc_id_badge = (
                            f'<span style="background: #F8F3FC; border: 1px solid #E5E7EB; color: #5A7383; '
                            f'font-size: 0.75rem; font-weight: 500; padding: 3px 10px; border-radius: 6px; font-family: monospace;">'
                            f'ID: {target_doc_id}</span>'
                            if target_doc_id else ''
                        )

                        st.markdown(
                            clean_html(f"""
                            <div class="document-metadata-box" style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 1px 4px rgba(27,69,88,0.03);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                                    <div style="display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.05rem; color: #1B4558;">
                                        {get_svg_icon("file-text", color="#EF3837", size=20)}
                                        <span>{fname}</span>
                                    </div>
                                    <div>{st_pill}</div>
                                </div>
                                <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                                    <span style="background: #F3E8FF; border: 1px solid #DDB7FF; color: #6B21A8; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 6px;">Type: {doc_type}</span>
                                    <span style="background: #F8F3FC; border: 1px solid #E5E7EB; color: #1B4558; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 6px;">Machine: {machine}</span>
                                    <span style="background: #F8F3FC; border: 1px solid #E5E7EB; color: #1B4558; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 6px;">Version: {version}</span>
                                    <span style="background: #F8F3FC; border: 1px solid #E5E7EB; color: #1B4558; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 6px;">Chunks: {chunks}</span>
                                    {doc_id_badge}
                                </div>
                            </div>
                            """),
                            unsafe_allow_html=True
                        )

                        try:
                            doc_text = fetch_document_content(target_doc_id, machine=machine)
                        except Exception:
                            doc_text = "Document content unavailable."

                        prev_col1, prev_col2 = st.columns([1.5, 1])
                        with prev_col1:
                            text_len = len(doc_text) if doc_text and doc_text != "Document content unavailable." else 0
                            st.markdown(
                                clean_html(f"""
                                <div style="font-size: 0.725rem; font-weight: 700; color: #EF3837; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 6px; padding: 4px 0;">
                                    {get_svg_icon("document", color="#EF3837", size=14)}
                                    <span>DOCUMENT CONTENT PREVIEW</span>
                                    <span style="color: #5A7383; font-weight: 500; font-size: 0.7rem; text-transform: none;">({text_len} chars)</span>
                                </div>
                                """),
                                unsafe_allow_html=True
                            )
                        with prev_col2:
                            expand_preview = st.checkbox("Expand full view", key=f"exp_{target_doc_id}")

                        st.markdown(format_document_preview_html(doc_text, is_expanded=expand_preview), unsafe_allow_html=True)

                        # Supervisor Deletion Action
                        if user_role == "Supervisor":
                            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                            del_col_left, del_col_right = st.columns([3, 1.2])
                            with del_col_right:
                                confirm_del_key = f"confirm_del_{target_doc_id}"
                                if st.session_state.get(confirm_del_key):
                                    st.warning(f"Delete {fname}?")
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        if st.button("Confirm", key=f"yes_del_{target_doc_id}", type="primary"):
                                            try:
                                                doc_to_delete = str(d["doc_id"]) if d.get("doc_id") else None
                                                deleted = post_document_delete(doc_id=doc_to_delete, filename=fname, machine=machine)
                                                if deleted:
                                                    st.session_state[confirm_del_key] = False
                                                    st.success(f"Document {fname} deleted.")
                                                    st.rerun()
                                                else:
                                                    st.error("Deletion failed.")
                                            except Exception as del_err:
                                                st.error(f"Error: {del_err}")
                                    with c2:
                                        if st.button("Cancel", key=f"no_del_{target_doc_id}"):
                                            st.session_state[confirm_del_key] = False
                                            st.rerun()
                                else:
                                    if st.button("Delete Document", key=f"del_btn_{target_doc_id}", type="secondary"):
                                        st.session_state[confirm_del_key] = True
                                        st.rerun()

                except Exception:
                    st.warning(f"Could not load preview for document: {d.get('filename', 'Unknown')}")