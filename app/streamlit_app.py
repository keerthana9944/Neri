import sys
import textwrap
from pathlib import Path
import requests
import streamlit as st

# ==================================================
# Project Setup & Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAG_API_URL
from src.history import auto_seed_if_empty

# Ensure all files under data/ are automatically seeded & indexed on startup
try:
    auto_seed_if_empty()
except Exception as e:
    print(f"Auto-seed note: {e}")

# Helper functions for API calls with direct Python fallback (Streamlit Cloud support)
def fetch_history_sessions():
    try:
        resp = requests.get(f"{RAG_API_URL}/history", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("sessions", [])
    except Exception:
        pass
    from src.history import get_history
    return get_history()

def fetch_documents_list():
    try:
        resp = requests.get(f"{RAG_API_URL}/documents", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    from src.history import get_documents
    return get_documents()

def fetch_session_detail(session_id: int):
    try:
        resp = requests.get(f"{RAG_API_URL}/history/{session_id}", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    from src.history import get_session
    return get_session(session_id)

def post_feedback(session_id: int, feedback: str, comment: str = None):
    try:
        resp = requests.post(
            f"{RAG_API_URL}/history/{session_id}/feedback",
            json={"feedback": feedback, "comment": comment},
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
        resp = requests.post(f"{RAG_API_URL}/query", json=payload, timeout=60)
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

def post_document_upload(file_name: str, file_bytes: bytes, document_type: str, machine: str = None, version: str = None, owner: str = None):
    try:
        files = {"file": (file_name, file_bytes)}
        data = {
            "document_type": document_type,
            "machine": machine,
            "version": version,
            "owner": owner
        }
        res = requests.post(f"{RAG_API_URL}/documents/upload", files=files, data=data, timeout=120)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    
    from src.config import UPLOAD_DIR
    from src.ingestion import create_chunks
    from src.embeddings import create_embeddings
    from src.vector_store import add_documents
    from src.history import save_document

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / Path(file_name).name
    file_path.write_bytes(file_bytes)

    chunks = create_chunks(
        file_path,
        document_type=document_type,
        machine=machine,
        version=version,
        owner=owner,
    )
    if not chunks:
        raise ValueError("No readable text was found in the document.")

    embeddings = create_embeddings([c["text"] for c in chunks])
    add_documents(chunks=chunks, embeddings=embeddings)
    save_document(
        filename=file_name,
        document_type=document_type,
        machine=machine,
        version=version,
        owner=owner,
        chunks=len(chunks),
        status="indexed",
    )
    return {
        "success": True,
        "message": "Document uploaded and indexed successfully.",
        "document": file_name,
        "document_type": document_type,
        "machine": machine,
        "version": version,
        "owner": owner,
        "chunks": len(chunks),
        "status": "indexed",
    }

def fetch_document_content(filename: str) -> str:
    try:
        resp = requests.get(f"{RAG_API_URL}/documents/{filename}/content", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("content", "")
    except Exception:
        pass
    from src.history import get_document_content
    return get_document_content(filename)

def clean_html(html_str: str) -> str:
    """Removes leading and trailing whitespace from each line to prevent Streamlit Markdown from treating HTML as code blocks."""
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "".join(lines)

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
    if query_view and query_view.lower() in valid_views and query_page != "login":
        st.session_state["logged_in"] = True
        st.session_state["show_landing_page"] = False
        target_role = "Supervisor" if (query_role and query_role.lower() == "supervisor") else "Technician"
        st.session_state["user_role"] = target_role
        st.session_state["username"] = "supervisor" if target_role == "Supervisor" else "technician"
        st.session_state["main_navigation_radio"] = valid_views[query_view.lower()]
    else:
        st.session_state["logged_in"] = False
        st.session_state["show_landing_page"] = False if query_page == "login" else True
        st.session_state["user_role"] = None
        st.session_state["username"] = None

if "show_landing_page" not in st.session_state:
    st.session_state["show_landing_page"] = False if query_page == "login" else True

if "user_role" not in st.session_state:
    st.session_state["user_role"] = None

if "username" not in st.session_state:
    st.session_state["username"] = None

if "troubleshooting_result" not in st.session_state:
    st.session_state["troubleshooting_result"] = None

if "history_detail" not in st.session_state:
    st.session_state["history_detail"] = None

if "show_feedback_comment" not in st.session_state:
    st.session_state["show_feedback_comment"] = False

if "history_filter_machine" not in st.session_state:
    st.session_state["history_filter_machine"] = None

# ==================================================
# SVG Vector Icon Helper System (STRICT: NO EMOJIS)
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
        "thumb-up": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>',
        "thumb-down": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>',
        "arrow-right": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>',
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "filter": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "server": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>',
        "clock": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        "file-text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
        "factory": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8l-7 5V8l-7 5V4H2v16z"/><line x1="17" y1="18" x2="17.01" y2="18"/><line x1="12" y1="18" x2="12.01" y2="18"/><line x1="7" y1="18" x2="7.01" y2="18"/></svg>',
        "chevron-right": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>',
        "menu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>'
    }
    return icons.get(name, f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>')


# ==================================================
# NERI DESIGN SYSTEM CSS (Cyber Industrial Blue & Glassmorphism)
# ==================================================

st.markdown(
    clean_html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulseGlow {
            0% {
                box-shadow: 0 0 0 0 rgba(14, 165, 233, 0.6);
            }
            70% {
                box-shadow: 0 0 0 10px rgba(14, 165, 233, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(14, 165, 233, 0);
            }
        }

        @keyframes borderGlow {
            0% { border-color: rgba(14, 165, 233, 0.2); }
            50% { border-color: rgba(14, 165, 233, 0.8); }
            100% { border-color: rgba(14, 165, 233, 0.2); }
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F172A;
            -webkit-font-smoothing: antialiased;
        }

        h1, h2, h3, h4, .neri-card-header {
            font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
        }

        .stApp {
            background: radial-gradient(circle at 50% -20%, #E0F2FE 0%, #F1F5F9 55%, #F8FAFC 100%) !important;
            background-attachment: fixed !important;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1240px;
            animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) ease-out;
        }

        /* Streamlit Primary Buttons — Gradient Cyber Blue #0284C7 -> #06B6D4 */
        div.stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%) !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 10px 22px !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            letter-spacing: 0.2px;
        }

        div.stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background: linear-gradient(135deg, #0369A1 0%, #0891B2 100%) !important;
            box-shadow: 0 8px 22px rgba(6, 182, 212, 0.45) !important;
            transform: translateY(-2px) scale(1.01);
        }

        /* Streamlit Form Input Styling */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea[data-testid="stTextArea"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            color: #0F172A !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.2s ease-in-out !important;
        }

        div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="input"]:focus-within > div,
        textarea[data-testid="stTextArea"]:focus {
            border-color: #0284C7 !important;
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.18) !important;
        }

        /* STRICT FIXED SIDEBAR SYSTEM (NO FLEXIBLE RESIZING) */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div:first-child,
        div[data-testid="stSidebarNav"] {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
            background: linear-gradient(180deg, #070D15 0%, #0F172A 100%) !important;
            border-right: 1px solid #1E293B !important;
            box-sizing: border-box !important;
        }

        /* Completely Hide Scrollbars on Sidebar */
        section[data-testid="stSidebar"]::-webkit-scrollbar,
        section[data-testid="stSidebar"] *::-webkit-scrollbar,
        section[data-testid="stSidebar"] > div::-webkit-scrollbar,
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]::-webkit-scrollbar {
            display: none !important;
            width: 0px !important;
            height: 0px !important;
            background: transparent !important;
        }

        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            -ms-overflow-style: none !important;
            scrollbar-width: none !important;
        }

        /* Disable Streamlit drag-to-resize handle completely */
        [data-testid="stSidebarResizer"] {
            display: none !important;
            width: 0px !important;
            pointer-events: none !important;
            visibility: hidden !important;
        }

        /* Prevent inner element horizontal overflow or clipping */
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
        section[data-testid="stSidebar"] div.stElementContainer {
            padding-left: 0px !important;
            padding-right: 0px !important;
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
        }

        section[data-testid="stSidebar"] *, 
        section[data-testid="stSidebar"] .stMarkdown, 
        section[data-testid="stSidebar"] label {
            color: #F8FAFC !important;
        }

        /* Remove Streamlit default white outlined button boxes in sidebar */
        section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: transparent;
            border: none;
            box-shadow: none;
            outline: none;
            box-sizing: border-box;
        }

        /* Streamlit Header & Sidebar Toggle Controls Styling */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99999 !important;
        }

        #MainMenu, footer {
            visibility: hidden !important;
            height: 0px !important;
        }

        /* Streamlit Collapsed Sidebar Control */
        [data-testid="stSidebarCollapsedControl"],
        button[data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarHeaderExpandButton"] {
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            z-index: 999999 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 38px !important;
            height: 38px !important;
            background-color: #0F172A !important;
            color: #38BDF8 !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stSidebarCollapsedControl"]:hover,
        button[data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="stSidebarHeaderExpandButton"]:hover {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border-color: #0284C7 !important;
        }

        /* Custom Cards & Containers with Animation & Scale Hover */
        .neri-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            animation: fadeInUp 0.4s ease-out;
        }

        .neri-card:hover {
            box-shadow: 0 12px 30px -4px rgba(15, 23, 42, 0.1);
            border-color: #BAE6FD;
            transform: translateY(-2px);
        }

        .neri-card-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Safety Warning Box (Amber Panel) */
        .warning-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #FCD34D;
            background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
            color: #92400E;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(245, 158, 11, 0.12);
            animation: fadeInUp 0.4s ease-out;
        }

        .warning-box-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #B45309;
        }

        /* Safety Critical Refusal Box (Red Panel) */
        .critical-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #FCA5A5;
            background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
            color: #991B1B;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(220, 38, 38, 0.12);
            animation: fadeInUp 0.4s ease-out;
        }

        .critical-box-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #DC2626;
        }

        .success-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #86EFAC;
            background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
            color: #14532D;
            margin-bottom: 20px;
        }

        .source-box {
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            background-color: #FFFFFF;
            margin-bottom: 12px;
            transition: all 0.25s ease;
        }

        .source-box:hover {
            border-color: #38BDF8;
            transform: translateX(4px);
        }

        .document-box {
            padding: 18px 20px;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            background-color: #FFFFFF;
            margin-bottom: 12px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
            transition: all 0.25s ease;
        }

        .document-box:hover {
            border-color: #0284C7;
            box-shadow: 0 6px 18px rgba(2, 132, 199, 0.08);
        }

        /* Numbered Badges & Workflow Steps */
        .neri-step-item {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 14px 0;
            border-bottom: 1px solid #F1F5F9;
            transition: all 0.2s ease;
        }

        .neri-step-item:hover {
            background-color: #F8FAFC;
            padding-left: 8px;
            border-radius: 8px;
        }

        .neri-badge-num {
            background: linear-gradient(135deg, #0F172A 0%, #0284C7 100%);
            color: #FFFFFF;
            font-size: 0.85rem;
            font-weight: 800;
            min-width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 3px 10px rgba(2, 132, 199, 0.25);
        }

        /* Scope main content secondary buttons */
        section[data-testid="stMain"] .stButton button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }

        section[data-testid="stMain"] .stButton button[kind="secondary"]:hover {
            border-color: #0284C7 !important;
            background-color: #F0F9FF !important;
            color: #0284C7 !important;
            transform: translateY(-1px);
        }

        /* Status Pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 0.775rem;
            font-weight: 700;
            background-color: #F0FDF4;
            color: #15803D;
            border: 1px solid #BBF7D0;
        }

        .status-pill.active {
            background-color: #F0F9FF;
            color: #0284C7;
            border: 1px solid #BAE6FD;
        }

        .status-pill.warning {
            background-color: #FFFBEB;
            color: #B45309;
            border: 1px solid #FDE68A;
        }

        .status-pill.error {
            background-color: #FEF2F2;
            color: #B91C1C;
            border: 1px solid #FCA5A5;
        }

        /* Streamlit Expander Styling */
        .streamlit-expanderHeader {
            font-weight: 700 !important;
            border-radius: 10px !important;
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            transition: all 0.2s ease !important;
        }

        .streamlit-expanderHeader:hover {
            border-color: #0284C7 !important;
            color: #0284C7 !important;
            background-color: #F0F9FF !important;
        }
    </style>
    """),
    unsafe_allow_html=True,
)

# ==================================================
# PUBLIC LANDING PAGE (Enterprise Product Introduction)
# ==================================================

if not st.session_state["logged_in"] and st.session_state.get("show_landing_page", True):

    # Top Navigation Bar
    nav_col1, nav_col2 = st.columns([1.5, 1])

    with nav_col1:
        st.markdown(
            clean_html(f"""
            <div style="display: flex; align-items: center; gap: 24px; margin-top: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    {get_svg_icon("logo", color="#06B6D4", size=26)}
                    <span style="font-size: 1.35rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px;">NERI</span>
                    <span style="font-size: 0.8rem; font-weight: 600; color: #64748B; border-left: 1px solid #CBD5E1; padding-left: 10px;">Maintenance Intelligence</span>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with nav_col2:
        _, btn_col2 = st.columns([1.5, 1])
        with btn_col2:
            if st.button("Sign In", type="primary", use_container_width=True, key="landing_top_signin"):
                st.session_state["show_landing_page"] = False
                st.query_params.clear()
                st.query_params["page"] = "login"
                st.rerun()

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Hero Section Banner (Dark Cyber Industrial Theme)
    st.markdown(
        clean_html(f"""
        <div style="background: linear-gradient(135deg, #070D15 0%, #0F172A 100%); border-radius: 16px; padding: 48px 40px; margin-bottom: 40px; border: 1px solid #1E293B; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);">
            <div style="max-width: 780px;">
                <div style="font-size: 0.775rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
                    {get_svg_icon("shield-check", color="#06B6D4", size=16)} AI MAINTENANCE INTELLIGENCE
                </div>
                <h1 style="font-size: 2.75rem; font-weight: 800; color: #FFFFFF; line-height: 1.15; letter-spacing: -0.02em; margin-bottom: 18px;">
                    Resolve machine problems<br><span style="background: linear-gradient(135deg, #38BDF8 0%, #06B6D4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">with grounded intelligence.</span>
                </h1>
                <p style="font-size: 1.05rem; line-height: 1.6; color: #94A3B8; margin-bottom: 28px; max-width: 640px;">
                    Neri helps maintenance teams troubleshoot industrial equipment using approved machine manuals, maintenance records, and safety procedures.
                </p>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    # Product Technical Diagnostic Preview Card (Horizontal Grid)
    st.markdown(
        clean_html(f"""
        <div style="margin-bottom: 40px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                TECHNICAL DIAGNOSTIC PREVIEW
            </div>
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 28px; box-shadow: 0 4px 20px -4px rgba(15, 23, 42, 0.06);">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F5F9; padding-bottom: 16px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        {get_svg_icon("wrench", color="#0F172A", size=22)}
                        <div>
                            <span style="font-weight: 800; font-size: 1.1rem; color: #0F172A;">Hydraulic Press</span>
                            <span style="font-size: 0.85rem; color: #64748B; margin-left: 8px; font-weight: 600;">ID: PRS-001</span>
                        </div>
                    </div>
                    <span class="status-pill active">
                        {get_svg_icon("shield-check", color="#0284C7", size=14)} Grounded in Documentation
                    </span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
                    <div style="background-color: #F0F9FF; border-radius: 10px; padding: 16px; border-left: 4px solid #0284C7;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Observed Problem</div>
                        <div style="font-size: 0.9rem; font-weight: 700; color: #0F172A; margin-top: 4px;">Pressure dropping under load</div>
                    </div>

                    <div style="background-color: #FFF7E6; border-radius: 10px; padding: 16px; border: 1px solid #F59E0B; color: #92400E;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #B45309; text-transform: uppercase; display: flex; align-items: center; gap: 6px;">
                            {get_svg_icon("alert", color="#B45309", size=14)} Safety Warning
                        </div>
                        <div style="font-size: 0.825rem; margin-top: 4px; line-height: 1.4;">
                            Depressurize accumulator before opening relief valves.
                        </div>
                    </div>

                    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px;">
                        <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; display: flex; align-items: center; gap: 6px;">
                            {get_svg_icon("document", color="#0284C7", size=14)} Source Reference
                        </div>
                        <div style="font-size: 0.825rem; color: #0F172A; font-weight: 600; margin-top: 4px;">Hydraulic System Manual v2.1</div>
                        <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Section 4.2 &bull; Page 34</div>
                    </div>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    # Value Capabilities Strip (4 Columns)
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)

    with v_col1:
        st.markdown(
            clean_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; height: 100%; transition: all 0.25s ease;" class="neri-card">
                {get_svg_icon("document", color="#0284C7", size=22)}
                <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 10px 0 4px 0;">GROUNDED</div>
                <div style="font-size: 0.825rem; color: #64748B; line-height: 1.45;">Answers are based on approved maintenance documentation.</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with v_col2:
        st.markdown(
            clean_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; height: 100%; transition: all 0.25s ease;" class="neri-card">
                {get_svg_icon("shield-check", color="#0284C7", size=22)}
                <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 10px 0 4px 0;">SAFETY-AWARE</div>
                <div style="font-size: 0.825rem; color: #64748B; line-height: 1.45;">Safety procedures remain part of the troubleshooting workflow.</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with v_col3:
        st.markdown(
            clean_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; height: 100%; transition: all 0.25s ease;" class="neri-card">
                {get_svg_icon("search", color="#0284C7", size=22)}
                <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 10px 0 4px 0;">TRACEABLE</div>
                <div style="font-size: 0.825rem; color: #64748B; line-height: 1.45;">Responses are connected to explicit source references.</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with v_col4:
        st.markdown(
            clean_html(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; height: 100%; transition: all 0.25s ease;" class="neri-card">
                {get_svg_icon("factory", color="#0284C7", size=22)}
                <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 10px 0 4px 0;">BUILT FOR MAINTENANCE</div>
                <div style="font-size: 0.825rem; color: #64748B; line-height: 1.45;">Designed around real manufacturing troubleshooting workflows.</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # Final Action CTA
    st.markdown(
        clean_html(f"""
        <div style="background: linear-gradient(135deg, #070D15 0%, #0F172A 100%); color: #FFFFFF; border-radius: 14px; padding: 40px; text-align: center; margin-bottom: 36px; border: 1px solid #1E293B;">
            <h2 style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; margin: 0 0 10px 0;">
                Ready to troubleshoot?
            </h2>
            <p style="font-size: 0.975rem; color: #94A3B8; max-width: 540px; margin: 0 auto 24px auto; line-height: 1.5;">
                Sign in to Neri Maintenance Intelligence to begin diagnosing shop floor machinery.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    cta_center_col1, cta_center_col2, cta_center_col3 = st.columns([1, 1.2, 1])
    with cta_center_col2:
        if st.button("Sign In to Neri", type="primary", use_container_width=True, key="landing_final_cta"):
            st.session_state["show_landing_page"] = False
            st.query_params.clear()
            st.query_params["page"] = "login"
            st.rerun()

    # Footer
    st.markdown(
        clean_html(f"""
        <div style="text-align: center; border-top: 1px solid #E2E8F0; padding-top: 24px; margin-top: 48px; color: #64748B; font-size: 0.825rem;">
            <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 6px;">
                {get_svg_icon("logo", color="#0284C7", size=18)}
                <strong style="color: #0F172A;">NERI Maintenance Intelligence</strong>
            </div>
            Authorized Industrial Operations Platform. Grounded in approved plant documentation.
        </div>
        """),
        unsafe_allow_html=True
    )

    st.stop()


# ==================================================
# LOGIN PAGE (Single Centered Enterprise Card)
# ==================================================

if not st.session_state["logged_in"]:

    st.markdown(
        clean_html("""
        <style>
        /* Turn the entire Login Column into ONE SINGLE UNIFIED WHITE CARD */
        div[data-testid="column"]:has(button[key="quick_tech_login"]) {
            background-color: #FFFFFF !important;
            border: 1px solid #D9E2E7 !important;
            border-radius: 16px !important;
            padding: 36px 32px !important;
            box-shadow: 0 8px 30px -4px rgba(11, 23, 38, 0.08) !important;
            max-width: 500px !important;
            margin: 0 auto !important;
        }

        /* Completely strip Streamlit's inner form border, padding, and box-shadow */
        div[data-testid="column"]:has(button[key="quick_tech_login"]) [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            margin: 0 !important;
        }

        /* Input styling inside the single login card */
        div[data-testid="column"]:has(button[key="quick_tech_login"]) [data-testid="stForm"] input {
            background-color: #F4F7F8 !important;
            border: 1px solid #D9E2E7 !important;
            border-radius: 8px !important;
            color: #14202B !important;
        }

        /* Access As Segmented Role Buttons Styling */
        div[data-testid="column"]:has(button[key="quick_tech_login"]) div.stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        </style>
        <div style="height: 36px;"></div>
        """),
        unsafe_allow_html=True
    )

    _, login_container, _ = st.columns([1, 1.2, 1])

    with login_container:
        # 1. Branding Header & Welcome Subtitle
        st.markdown(
            clean_html(f"""
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="display: inline-flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 16px;">
                    {get_svg_icon("logo", color="#06B6D4", size=32)}
                    <div style="text-align: left;">
                        <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; line-height: 1.1;">NERI</div>
                        <div style="font-size: 0.775rem; font-weight: 600; color: #0284C7;">Maintenance Intelligence</div>
                    </div>
                </div>
                <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 0 0 4px 0;">Welcome back</h3>
                <div style="font-size: 0.875rem; color: #64748B;">Sign in to Neri Maintenance Intelligence</div>
            </div>
            """),
            unsafe_allow_html=True
        )

        # 2. Access As Segmented Control Header
        st.markdown(
            clean_html("""
            <div style="margin-bottom: 8px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">ACCESS AS</div>
            </div>
            """),
            unsafe_allow_html=True
        )

        # Single source of truth for selected role
        if "login_selected_role" not in st.session_state:
            st.session_state["login_selected_role"] = "Technician"

        selected_role = st.session_state["login_selected_role"]

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            tech_type = "primary" if selected_role == "Technician" else "secondary"
            if st.button("Technician", type=tech_type, use_container_width=True, key="quick_tech_login"):
                st.session_state["login_selected_role"] = "Technician"
                st.rerun()

        with d_col2:
            super_type = "primary" if selected_role == "Supervisor" else "secondary"
            if st.button("Supervisor", type=super_type, use_container_width=True, key="quick_super_login"):
                st.session_state["login_selected_role"] = "Supervisor"
                st.rerun()

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        username_default = "technician" if selected_role == "Technician" else "supervisor"

        # 3. Form Section (Inner stForm border stripped by CSS above)
        with st.form("login_form"):
            username_input = st.text_input("Username", value=username_default, placeholder="Enter your username")
            password_input = st.text_input("Password", type="password", placeholder="Enter your password")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit_login:
                username_clean = username_input.strip().lower()
                password_clean = password_input.strip()

                if username_clean == "technician" and password_clean == "tech123" and selected_role == "Technician":
                    st.session_state["logged_in"] = True
                    st.session_state["user_role"] = "Technician"
                    st.session_state["username"] = "technician"
                    st.session_state["show_landing_page"] = False
                    st.session_state["main_navigation_radio"] = "Dashboard"
                    st.query_params.clear()
                    st.query_params["view"] = "dashboard"
                    st.query_params["role"] = "technician"
                    st.rerun()

                elif username_clean == "supervisor" and password_clean == "super123" and selected_role == "Supervisor":
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
                    st.error("Invalid username or password.")

        # 4. Footer inside the Single Card
        st.markdown(
            clean_html("""
            <div style="text-align: center; font-size: 0.775rem; color: #64748B; margin-top: 20px;">
                Authorized maintenance personnel
            </div>
            """),
            unsafe_allow_html=True
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

def nav_to_history(machine_filter: str = None):
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
    st.session_state["history_detail"] = None
    st.session_state["show_feedback_comment"] = False
    st.session_state["main_navigation_radio"] = "Dashboard"
    st.query_params.clear()
    st.query_params["page"] = "login"

user_role = st.session_state.get("user_role", "Technician")
username = st.session_state.get("username", "technician")

# Sidebar Navigation (Redesigned Enterprise Sidebar in #0B1726)
with st.sidebar:
    # 1. Glassmorphism Branding Header
    st.markdown(
        clean_html(f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(12px); border-radius: 12px; padding: 14px 16px; margin-bottom: 22px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.35); border-radius: 10px; padding: 7px; display: flex; align-items: center; justify-content: center;">
                    {get_svg_icon("logo", color="#06B6D4", size=24)}
                </div>
                <div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; line-height: 1.1;">NERI</div>
                    <div style="font-size: 0.7rem; font-weight: 600; color: #38BDF8; margin-top: 2px;">Maintenance AI</div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 5px; background: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 20px; padding: 3px 8px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background-color: #06B6D4; box-shadow: 0 0 8px #06B6D4; animation: pulseGlow 2s infinite;"></span>
                <span style="font-size: 0.65rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.5px;">ONLINE</span>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    nav_options = ["Dashboard", "Troubleshoot", "History", "Documents"]

    # Sync navigation state with URL query parameters on initial load / refresh
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

    # SVG Icon URIs for CSS styling
    svg_data_uris = {
        "Dashboard": {
            "active": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='9'/%3E%3Crect x='14' y='3' width='7' height='5'/%3E%3Crect x='14' y='12' width='7' height='9'/%3E%3Crect x='3' y='15' width='7' height='6'/%3E%3C/svg%3E",
            "inactive": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238EA1AE' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='7' height='9'/%3E%3Crect x='14' y='3' width='7' height='5'/%3E%3Crect x='14' y='12' width='7' height='9'/%3E%3Crect x='3' y='15' width='7' height='6'/%3E%3C/svg%3E"
        },
        "Troubleshoot": {
            "active": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z'/%3E%3C/svg%3E",
            "inactive": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238EA1AE' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z'/%3E%3C/svg%3E"
        },
        "History": {
            "active": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E",
            "inactive": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238EA1AE' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpolyline points='12 6 12 12 16 14'/%3E%3C/svg%3E"
        },
        "Documents": {
            "active": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3C/svg%3E",
            "inactive": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238EA1AE' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3C/svg%3E"
        }
    }

    sidebar_css = [
        """
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070D15 0%, #0F172A 100%) !important;
            border-right: 1px solid #1E293B !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: transparent !important;
            color: #94A3B8 !important;
            border: none !important;
            border-radius: 8px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            padding: 10px 14px 10px 42px !important;
            position: relative !important;
            margin-bottom: 6px !important;
            box-shadow: none !important;
            outline: none !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            transform: translateX(3px) !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover *,
        section[data-testid="stSidebar"] div.stButton > button:hover p,
        section[data-testid="stSidebar"] div.stButton > button:hover span {
            color: #FFFFFF !important;
            background: transparent !important;
        }
        """
    ]

    for item in nav_options:
        is_active = (item == selected_nav)
        icon_url = svg_data_uris[item]["active"] if is_active else svg_data_uris[item]["inactive"]
        
        rule = f"""
        section[data-testid="stSidebar"] div#nav_btn_{item} button,
        section[data-testid="stSidebar"] div[id*="nav_btn_{item}"] button,
        section[data-testid="stSidebar"] button[key="nav_btn_{item}"] {{
            background: {"linear-gradient(90deg, rgba(2, 132, 199, 0.25) 0%, rgba(15, 23, 42, 0.6) 100%)" if is_active else "transparent"} !important;
            color: {"#FFFFFF" if is_active else "#94A3B8"} !important;
            border-left: {"5px solid #06B6D4" if is_active else "5px solid transparent"} !important;
            font-weight: {"800" if is_active else "600"} !important;
            box-shadow: {"0 4px 14px rgba(2, 132, 199, 0.3)" if is_active else "none"} !important;
        }}
        section[data-testid="stSidebar"] div#nav_btn_{item} button:active,
        section[data-testid="stSidebar"] div#nav_btn_{item} button:focus,
        section[data-testid="stSidebar"] div[id*="nav_btn_{item}"] button:active,
        section[data-testid="stSidebar"] div[id*="nav_btn_{item}"] button:focus,
        section[data-testid="stSidebar"] button[key="nav_btn_{item}"]:active,
        section[data-testid="stSidebar"] button[key="nav_btn_{item}"]:focus {{
            background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%) !important;
            color: #FFFFFF !important;
        }}
        section[data-testid="stSidebar"] div#nav_btn_{item} button *,
        section[data-testid="stSidebar"] div[id*="nav_btn_{item}"] button *,
        section[data-testid="stSidebar"] button[key="nav_btn_{item}"] * {{
            color: {"#FFFFFF" if is_active else "#94A3B8"} !important;
            background: transparent !important;
        }}
        section[data-testid="stSidebar"] div#nav_btn_{item} button::before,
        section[data-testid="stSidebar"] div[id*="nav_btn_{item}"] button::before,
        section[data-testid="stSidebar"] button[key="nav_btn_{item}"]::before {{
            content: "" !important;
            position: absolute !important;
            left: 14px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 18px !important;
            height: 18px !important;
            background-image: url("{icon_url}") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            filter: {"drop-shadow(0 0 6px rgba(255, 255, 255, 0.6))" if is_active else "none"} !important;
        }}
        """
        sidebar_css.append(rule)

    # Logout button styling — Solid White Box Background with Red Text & Red Vector Icon
    logout_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23DC2626' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4'/%3E%3Cpolyline points='16 17 21 12 16 7'/%3E%3Cline x1='21' y1='12' x2='9' y2='12'/%3E%3C/svg%3E"
    sidebar_css.append(f"""
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button,
    section[data-testid="stSidebar"] div#logout_btn button,
    section[data-testid="stSidebar"] div[id*="logout"] button,
    section[data-testid="stSidebar"] button[key="logout_btn"] {{
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #DC2626 !important;
        border: 1px solid #D9E2E7 !important;
        border-radius: 8px !important;
        text-align: left !important;
        padding: 10px 14px 10px 42px !important;
        position: relative !important;
        font-size: 0.875rem !important;
        font-weight: 700 !important;
        margin-top: 14px !important;
        width: 100% !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.2s ease !important;
    }}
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button:hover,
    section[data-testid="stSidebar"] div#logout_btn button:hover,
    section[data-testid="stSidebar"] div[id*="logout"] button:hover,
    section[data-testid="stSidebar"] button[key="logout_btn"]:hover,
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button:focus,
    section[data-testid="stSidebar"] div#logout_btn button:focus,
    section[data-testid="stSidebar"] div[id*="logout"] button:focus,
    section[data-testid="stSidebar"] button[key="logout_btn"]:focus {{
        background-color: #FEE2E2 !important;
        background: #FEE2E2 !important;
        color: #B91C1C !important;
        border-color: #FCA5A5 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.25) !important;
    }}
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button *,
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button p,
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button span,
    section[data-testid="stSidebar"] div#logout_btn button *,
    section[data-testid="stSidebar"] div[id*="logout"] button *,
    section[data-testid="stSidebar"] button[key="logout_btn"] *,
    section[data-testid="stSidebar"] div#logout_btn button p,
    section[data-testid="stSidebar"] div[id*="logout"] button p,
    section[data-testid="stSidebar"] button[key="logout_btn"] p,
    section[data-testid="stSidebar"] div#logout_btn button span,
    section[data-testid="stSidebar"] div[id*="logout"] button span,
    section[data-testid="stSidebar"] button[key="logout_btn"] span {{
        color: #DC2626 !important;
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button:hover *,
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button:hover p,
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button:hover span,
    section[data-testid="stSidebar"] div#logout_btn button:hover *,
    section[data-testid="stSidebar"] div[id*="logout"] button:hover *,
    section[data-testid="stSidebar"] button[key="logout_btn"]:hover *,
    section[data-testid="stSidebar"] div#logout_btn button:hover p,
    section[data-testid="stSidebar"] div[id*="logout"] button:hover p,
    section[data-testid="stSidebar"] button[key="logout_btn"]:hover p,
    section[data-testid="stSidebar"] div#logout_btn button:hover span,
    section[data-testid="stSidebar"] div[id*="logout"] button:hover span,
    section[data-testid="stSidebar"] button[key="logout_btn"]:hover span {{
        color: #B91C1C !important;
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] div.stElementContainer:has(#logout-marker) + div.stElementContainer button::before,
    section[data-testid="stSidebar"] div#logout_btn button::before,
    section[data-testid="stSidebar"] div[id*="logout"] button::before,
    section[data-testid="stSidebar"] button[key="logout_btn"]::before {{
        content: "" !important;
        position: absolute !important;
        left: 14px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 16px !important;
        height: 16px !important;
        background-image: url("{logout_icon_url}") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
    }}
    """)

    st.markdown(
        clean_html(f"""
        <style>
        {' '.join(sidebar_css)}
        </style>
        """),
        unsafe_allow_html=True
    )

    # Render Sidebar Navigation Items
    for item in nav_options:
        btn_key = f"nav_btn_{item}"
        if st.button(item, key=btn_key, use_container_width=True, type="secondary"):
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

    # 3. Glassmorphic User Profile Card at Bottom of Sidebar
    role_badge_color = "#06B6D4" if user_role == "Supervisor" else "#38BDF8"
    role_badge_bg = "rgba(6, 182, 212, 0.15)" if user_role == "Supervisor" else "rgba(56, 189, 248, 0.15)"
    role_badge_border = "rgba(6, 182, 212, 0.35)" if user_role == "Supervisor" else "rgba(56, 189, 248, 0.35)"

    st.markdown(
        clean_html(f"""
        <div style="margin-top: 32px; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(6, 182, 212, 0.25); backdrop-filter: blur(10px); border-radius: 12px; padding: 14px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); padding: 7px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                        {get_svg_icon("user", color="#06B6D4", size=18)}
                    </div>
                    <div>
                        <div style="font-size: 0.875rem; font-weight: 700; color: #FFFFFF; line-height: 1.2;">{username}</div>
                        <div style="margin-top: 4px;">
                            <span style="font-size: 0.675rem; font-weight: 700; color: {role_badge_color}; background: {role_badge_bg}; border: 1px solid {role_badge_border}; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">
                                {user_role}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    # 4. Logout Action Button
    st.markdown('<div id="logout-marker"></div>', unsafe_allow_html=True)
    st.button("Logout", use_container_width=True, type="secondary", key="logout_btn", on_click=do_logout)

# Top Header Bar in Main Layout
st.markdown(
    clean_html(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid #E2E8F0; margin-bottom: 24px;">
        <div>
            <span style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">WORKSPACE</span>
            <h2 style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 0;">{selected_nav}</h2>
        </div>
        <div class="status-pill active">
            {get_svg_icon("server", color="#0284C7", size=14)} Knowledge Base Active
        </div>
    </div>
    """),
    unsafe_allow_html=True
)

# ==================================================
# PAGE 1: DASHBOARD (Operations Overview Layout)
# ==================================================

if selected_nav == "Dashboard":

    st.markdown(
        clean_html(f"""
        <div class="neri-card" style="border-left: 4px solid #0284C7;">
            <h3 style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;">Welcome back, {username}</h3>
            <p style="font-size: 0.875rem; color: #64748B; margin: 0;">
                Access grounded maintenance intelligence and troubleshooting assistance for plant machinery.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    # Fetch stats using unified helpers (supports Streamlit Cloud direct access)
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

    # 3 Clean Informational KPI Cards
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Active Machines</span>
                    {get_svg_icon("wrench", color="#0284C7", size=20)}
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #0F172A;">{len(unique_machines) if unique_machines else '—'}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Registered equipment in history</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m_col2:
        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Troubleshooting Sessions</span>
                    {get_svg_icon("history", color="#0284C7", size=20)}
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #0F172A;">{total_sessions_count}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Recorded diagnostic queries</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with m_col3:
        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Approved Documents</span>
                    {get_svg_icon("document", color="#0284C7", size=20)}
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: #0F172A;">{approved_docs_count}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Indexed manuals, logs & safety specs</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Primary Operational Command Action Panel
    st.markdown(
        clean_html("""
        <div class="neri-card" style="background: #FFFFFF; border-left: 4px solid #0284C7; margin-top: 8px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; margin-bottom: 4px;">OPERATIONAL COMMAND</div>
            <h4 style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 0 0 6px 0;">Need to diagnose a machine problem?</h4>
            <p style="font-size: 0.875rem; color: #64748B; margin: 0 0 16px 0; line-height: 1.5;">
                Start a grounded troubleshooting session using approved manuals and maintenance documentation.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )
    
    st.button("Start Troubleshooting", type="primary", key="dash_start_tb", on_click=nav_to_troubleshoot)

    # Recent Troubleshooting Activity Table
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        clean_html(f"""
        <div class="neri-card-header">
            {get_svg_icon("clock", color="#0B1726", size=20)} Recent Activity
        </div>
        """),
        unsafe_allow_html=True
    )

    try:
        recent_sessions = fetch_history_sessions()[:5]
        if not recent_sessions:
            st.info("No recent troubleshooting sessions recorded.")
        else:
                for s in recent_sessions:
                    m_name = s.get("machine", "Unknown Machine")
                    m_id = s.get("machine_id", "N/A")
                    prob = s.get("problem", "N/A")
                    dt = s.get("created_at", "")[:16].replace("T", " ")
                    fb = s.get("feedback")
                    fb_badge = f'<span class="status-pill active">{fb.title()}</span>' if fb else '<span style="color:#5B7180; font-size:0.75rem;">No Feedback</span>'

                    st.markdown(
                        clean_html(f"""
                        <div class="document-box">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="color: #0B1726;">{m_name}</strong> 
                                    <span style="color: #5B7180; font-size: 0.8rem;">(ID: {m_id})</span>
                                    <div style="font-size: 0.825rem; color: #14202B; margin-top: 2px;">{prob[:90]}{'...' if len(prob) > 90 else ''}</div>
                                </div>
                                <div style="text-align: right; flex-shrink: 0;">
                                    <div style="font-size: 0.75rem; color: #5B7180; margin-bottom: 4px;">{dt}</div>
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
# PAGE 2: TROUBLESHOOTING (Diagnostic Workspace Layout)
# ==================================================

elif selected_nav == "Troubleshoot":

    st.markdown(
        clean_html("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.5px;">DIAGNOSTIC WORKSPACE</div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 2px 0 4px 0;">Diagnose a machine problem</h3>
            <p style="font-size: 0.875rem; color: #64748B; margin: 0;">
                Describe the issue and Neri will search approved documentation for grounded guidance.
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
                "What problem are you experiencing?",
                placeholder="Describe observed machine symptoms, leaks, pressure drops, unusual noises, etc...",
                height=120
            )

            error_code = st.text_input("Error Code (optional)", placeholder="e.g. E-115")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            submit_troubleshoot = st.form_submit_button("Analyze Problem", type="primary", use_container_width=True)

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

                    # Visual Progress Experience (Vector Icons Only)
                    progress_placeholder = st.empty()
                    progress_placeholder.markdown(
                        clean_html(f"""
                        <div class="neri-card" style="border-left: 4px solid #0284C7; padding: 20px;">
                            <div style="font-weight: 700; font-size: 1rem; color: #0F172A; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">
                                {get_svg_icon("cpu", color="#0284C7", size=20)} 
                                <span>Neri Diagnostic Pipeline Active</span>
                            </div>
                            <div style="font-size: 0.875rem; color: #64748B; line-height: 1.8; display: flex; flex-direction: column; gap: 6px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("search", color="#0284C7", size=14)} <span>Understanding problem & symptoms...</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("document", color="#0284C7", size=14)} <span>Searching approved documentation...</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("history", color="#0284C7", size=14)} <span>Checking maintenance records...</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("shield", color="#0284C7", size=14)} <span>Checking safety procedures...</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("check", color="#06B6D4", size=14)} <strong style="color: #0F172A;">Preparing troubleshooting guidance...</strong>
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
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px;">
                <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.95rem; color: #0F172A; margin-bottom: 12px;">
                    {get_svg_icon("shield-check", color="#0284C7", size=18)} Safety First & Tips
                </div>
                <div style="font-size: 0.825rem; color: #64748B; line-height: 1.5; display: flex; flex-direction: column; gap: 10px;">
                    <div><strong>1. Machine Identification:</strong> Include exact machine name and plant ID for precise retrieval.</div>
                    <div><strong>2. Detailed Symptoms:</strong> Mention leaks, noise type, temperature readings, or pressure drops.</div>
                    <div><strong>3. Error Codes:</strong> Enter control panel error codes if available (e.g. E-402).</div>
                    <div style="background: #FFFBEB; padding: 10px; border-radius: 8px; border: 1px solid #FCD34D; color: #92400E; margin-top: 4px;">
                        <strong>Plant Safety Notice:</strong> Follow LOTO requirements before inspecting electrical or pressure components.
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

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.divider()

        is_reliable = result.get("reliable", False)

        if not is_reliable:
            # NO RELIABLE ANSWER STATE / SAFETY REFUSAL
            safety_warning = result.get("safety_warning")
            message_text = result.get("message", "We couldn't find enough information in approved documentation to provide a reliable answer.")
            
            if safety_warning:
                # SAFETY-CRITICAL REFUSAL STATE
                st.markdown(
                    clean_html(f"""
                    <div class="critical-box">
                        <div class="critical-box-title">
                            {get_svg_icon("alert", color="#DC2626", size=22)} Safety Documentation Required
                        </div>
                        <div style="font-size: 0.9rem; font-weight: 500; line-height: 1.6; margin-bottom: 10px;">
                            {message_text}
                        </div>
                        <div style="font-size: 0.825rem; color: #7F1D1D; background: rgba(220, 38, 38, 0.08); padding: 10px; border-radius: 6px;">
                            Neri cannot provide troubleshooting steps for safety-critical components without verified safety documentation.
                        </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
            else:
                # NO RELIABLE ANSWER STATE
                st.markdown(
                    clean_html(f"""
                    <div class="neri-card" style="border-left: 4px solid #0284C7; background: #FFFFFF; padding: 24px;">
                        <div style="display: flex; align-items: center; gap: 10px; font-size: 1.1rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
                            {get_svg_icon("document", color="#0284C7", size=22)} No reliable answer found
                        </div>
                        <div style="font-size: 0.9rem; color: #64748B; line-height: 1.6; margin-bottom: 12px;">
                            {message_text}
                        </div>
                        <div style="font-size: 0.825rem; color: #0284C7; background: #F0F9FF; padding: 10px; border-radius: 6px; font-weight: 500;">
                            Try describing the machine symptoms differently or check whether the required documentation is available in the Knowledge Base.
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

            err_code_badge = f'<span style="background: #F1F5F9; color: #64748B; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">Code: {err_code}</span>' if err_code else ''

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-bottom: 16px; padding: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;">
                        <div>
                            <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.5px;">DIAGNOSTIC REPORT</div>
                            <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 4px 0 10px 0;">Troubleshooting Guidance</h3>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                                <span>{m_name}</span>
                                <span style="color: #CBD5E1;">•</span>
                                <span style="color: #64748B;">ID: {m_id}</span>
                                {err_code_badge}
                            </div>
                            <div style="font-size: 0.875rem; color: #64748B; margin-top: 6px; line-height: 1.4;">
                                <strong style="color: #0F172A;">Problem:</strong> {prob_txt}
                            </div>
                        </div>
                        <span class="status-pill active" style="flex-shrink: 0;">
                            {get_svg_icon("shield-check", color="#0284C7", size=14)} Grounded in Documentation
                        </span>
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

            # Safety Warning Box
            safety = result.get("safety_warning")
            if safety:
                safety_msg = safety.get("message", "") if isinstance(safety, dict) else str(safety)
                if safety_msg.strip():
                    st.markdown(
                        clean_html(f"""
                        <div class="warning-box" style="margin-bottom: 16px;">
                            <div class="warning-box-title">
                                {get_svg_icon("alert", color="#B45309", size=20)} Safety Warning
                            </div>
                            <div style="font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
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
                        <div style="font-size: 0.9rem; font-weight: 600; color: #0F172A; margin-top: 4px;">{cause_str}</div>
                    </div>
                    """)
                causes_body = "".join(causes_html)
            else:
                causes_body = '<div style="font-size: 0.875rem; color: #64748B; padding: 12px 0;">No documented causes found.</div>'

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-bottom: 16px;">
                    <div class="neri-card-header" style="border-bottom: 1px solid #F1F5F9; padding-bottom: 12px; margin-bottom: 8px;">
                        {get_svg_icon("search", color="#0F172A", size=18)} Possible Causes
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
                        <div style="font-size: 0.9rem; color: #0F172A; line-height: 1.5; margin-top: 4px;">{step_str}</div>
                    </div>
                    """)
                steps_body = "".join(steps_html)
            else:
                steps_body = '<div style="font-size: 0.875rem; color: #64748B; padding: 12px 0;">No documented troubleshooting steps found.</div>'

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-bottom: 16px;">
                    <div class="neri-card-header" style="border-bottom: 1px solid #F1F5F9; padding-bottom: 12px; margin-bottom: 8px;">
                        {get_svg_icon("wrench", color="#0F172A", size=18)} Recommended Troubleshooting Steps
                    </div>
                    {steps_body}
                </div>
                """),
                unsafe_allow_html=True
            )

            # Source References (Grouped by Document)
            sources = result.get("sources", [])
            if sources:
                grouped_sources = {}
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
                    
                    sec_list = sorted(list(info["sections"]))
                    sec_val = f"Section: {', '.join(sec_list)}" if sec_list else "Section: Not available"
                    
                    pg_list = sorted(list(info["pages"]))
                    pg_val = f"Page: {', '.join(pg_list)}" if pg_list else "Page: Not available"

                    src_cards_html.append(f"""
                    <div class="source-box" style="margin-bottom: 10px;">
                        <div style="font-weight: 700; font-size: 0.9rem; color: #0F172A; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                {get_svg_icon("file-text", color="#0284C7", size=18)}
                                <span>{doc_name}</span>
                            </div>
                            <span style="font-size: 0.725rem; font-weight: 700; color: #0284C7; background: #F0F9FF; padding: 2px 8px; border-radius: 4px;">{doc_type_str}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B; display: flex; gap: 16px;">
                            <span>{sec_val}</span>
                            <span>{pg_val}</span>
                            <span style="color: #0284C7; font-weight: 600;">{count_str}</span>
                        </div>
                    </div>
                    """)
                
                sources_body = "".join(src_cards_html)
            else:
                sources_body = '<div style="font-size: 0.875rem; color: #64748B; padding: 12px 0;">No source references available.</div>'

            st.markdown(
                clean_html(f"""
                <div class="neri-card" style="margin-top: 16px; margin-bottom: 16px; padding: 20px;">
                    <div class="neri-card-header" style="border-bottom: 1px solid #F1F5F9; padding-bottom: 12px; margin-bottom: 12px;">
                        {get_svg_icon("document", color="#0F172A", size=18)} Source References
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
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;">
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
# PAGE 3: HISTORY (Case Archive Layout)
# ==================================================

elif selected_nav == "History":

    st.markdown(
        clean_html("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.5px;">CASE ARCHIVE</div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 2px 0 4px 0;">Troubleshooting Case History</h3>
            <p style="font-size: 0.875rem; color: #64748B; margin: 0;">
                Review previous diagnostic sessions and archived technical reports.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    try:
        sessions = fetch_history_sessions()
        total_sessions = len(sessions)
        helpful_sessions = len([s for s in sessions if s.get("feedback") == "helpful"])
        unique_machines_count = len(set([s.get("machine") for s in sessions if s.get("machine")]))

        hk_col1, hk_col2, hk_col3 = st.columns(3)
        with hk_col1:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Total Sessions</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{total_sessions}</div>
            </div>
            """), unsafe_allow_html=True)
        with hk_col2:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Helpful Feedback</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #0284C7; margin-top: 4px;">{helpful_sessions}</div>
            </div>
            """), unsafe_allow_html=True)
        with hk_col3:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Unique Equipment</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{unique_machines_count}</div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        if not sessions:
            st.info("No troubleshooting sessions found in history.")
        else:
            # Optional filter clearing
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
                        <div style="font-size: 0.875rem; color: #0F172A; line-height: 1.6; margin-bottom: 12px;">
                            <strong>Date/Time:</strong> {dt}<br>
                            <strong>Machine:</strong> {m_name} ({m_id})<br>
                            <strong>Problem:</strong> {prob}<br>
                            {"<strong>Error Code:</strong> " + session['error_code'] + "<br>" if session.get('error_code') else ""}
                            <strong>Status:</strong> {fb_str}
                        </div>
                        """),
                        unsafe_allow_html=True
                    )

                    if st.button("View Archived Diagnostic Report", key=f"hist_view_{s_id}"):
                        try:
                            det = fetch_session_detail(s_id)
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

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.divider()

        st.markdown(
            clean_html(f"""
            <div class="neri-card" style="border-left: 4px solid #0284C7;">
                <div class="neri-card-header">
                    {get_svg_icon("document", color="#0284C7", size=20)} Archived Diagnostic Case Report
                </div>
                <div style="font-size: 0.875rem; color: #0F172A; line-height: 1.6;">
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
                    <div style="font-size: 0.875rem; font-weight: 600; color: #0F172A; margin-top: 4px;">{c_str}</div>
                </div>
                """)

            st.markdown(
                clean_html(f"""
                <div class="neri-card">
                    <div class="neri-card-header">{get_svg_icon("search", color="#0F172A", size=18)} Archived Causes</div>
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
                    <div style="font-size: 0.875rem; color: #0F172A; line-height: 1.5; margin-top: 4px;">{s_str}</div>
                </div>
                """)

            st.markdown(
                clean_html(f"""
                <div class="neri-card">
                    <div class="neri-card-header">{get_svg_icon("wrench", color="#0F172A", size=18)} Archived Steps</div>
                    {''.join(steps_markup)}
                </div>
                """),
                unsafe_allow_html=True
            )

        if st.button("Close Case Report", key="close_case_report"):
            st.session_state["history_detail"] = None
            st.rerun()


# ==================================================
# PAGE 4: DOCUMENTS (Knowledge Base Management Center)
# ==================================================

elif selected_nav == "Documents":

    st.markdown(
        clean_html("""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.5px;">KNOWLEDGE BASE CENTER</div>
            <h3 style="font-size: 1.35rem; font-weight: 800; color: #0F172A; margin: 2px 0 4px 0;">Approved Documentation Management</h3>
            <p style="font-size: 0.875rem; color: #64748B; margin: 0;">
                Manage approved manuals, maintenance logs, and safety procedures used by Neri.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    # Document Summary KPIs
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
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Total Documents</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{total_docs}</div>
            </div>
            """), unsafe_allow_html=True)
        with dk_col2:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Manuals</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #0284C7; margin-top: 4px;">{manuals_count}</div>
            </div>
            """), unsafe_allow_html=True)
        with dk_col3:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Logs</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #0F172A; margin-top: 4px;">{logs_count}</div>
            </div>
            """), unsafe_allow_html=True)
        with dk_col4:
            st.markdown(clean_html(f"""
            <div class="neri-card" style="padding: 16px; height: 100%;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Safety Specs</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #D97706; margin-top: 4px;">{safety_count}</div>
            </div>
            """), unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Role Authorization Check: Only Supervisors can Upload
    if user_role == "Supervisor":
        with st.expander("Add Approved Documentation", expanded=False):
            with st.form("doc_upload_form"):
                st.subheader("Upload Approved Documentation")

                doc_type = st.selectbox("Document Type", ["Manual", "Maintenance Log", "Safety Procedure"])

                uc1, uc2 = st.columns(2)
                with uc1:
                    machine_tag = st.text_input("Machine (Optional)", placeholder="e.g. CNC Mill")
                with uc2:
                    version_tag = st.text_input("Version (Optional)", placeholder="e.g. 1.0")

                owner_tag = st.text_input("Owner (Optional)", placeholder="e.g. Maintenance Dept")

                up_file = st.file_uploader("Select PDF or TXT file", type=["pdf", "txt"])

                submit_upload = st.form_submit_button("Upload & Index", type="primary", use_container_width=True)

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
                            else:
                                st.error("Upload failed.")

                        except Exception as e:
                            st.error(f"Error indexing document: {e}")

    # Search / Filter Bar
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        search_query = st.text_input("Search documents", placeholder="Filter by document name or machine...", key="doc_search_input")
    with f_col2:
        type_filter = st.selectbox("Filter Type", ["All Types", "Manual", "Maintenance Log", "Safety Procedure"])

    # Approved Documentation Directory List
    try:
        docs = fetch_documents_list()

        if search_query.strip():
            q = search_query.strip().lower()
            docs = [d for d in docs if q in (d.get("filename", "") or "").lower() or q in (d.get("machine", "") or "").lower() or q in (d.get("document_type", "") or "").lower()]

        if type_filter != "All Types":
            tf_slug = type_filter.lower().replace(" ", "_")
            docs = [d for d in docs if tf_slug in str(d.get("document_type", "")).lower() or type_filter.lower() in str(d.get("document_type", "")).lower()]

        if not docs:
            st.info("No matching documents found in knowledge base.")
        else:
            for d in docs:
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

                exp_label = f"{fname}  •  {doc_type}  •  {machine}  ({chunks} Chunks)"

                with st.expander(exp_label):
                    st.markdown(
                        clean_html(f"""
                        <div class="document-box" style="margin-bottom: 12px; background: #FFFFFF;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div style="font-weight: 700; font-size: 1rem; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                                    {get_svg_icon("file-text", color="#0284C7", size=20)} {fname}
                                </div>
                                {st_pill}
                            </div>
                            <div style="font-size: 0.825rem; color: #64748B; line-height: 1.5;">
                                <strong>Type:</strong> {doc_type} &bull; 
                                <strong>Machine:</strong> {machine} &bull; 
                                <strong>Version:</strong> {version} &bull; 
                                <strong>Chunks:</strong> {chunks}
                            </div>
                        </div>
                        <div style="font-size: 0.75rem; font-weight: 700; color: #0284C7; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                            {get_svg_icon("document", color="#0284C7", size=14)} DOCUMENT CONTENT PREVIEW
                        </div>
                        """),
                        unsafe_allow_html=True
                    )

                    doc_text = fetch_document_content(fname)

                    st.markdown(
                        clean_html(f"""
                        <div style="background: #0F172A; color: #E2E8F0; border-radius: 8px; padding: 18px; font-family: 'Consolas', 'Courier New', monospace; font-size: 0.85rem; line-height: 1.6; max-height: 420px; overflow-y: auto; white-space: pre-wrap; border: 1px solid #1E293B;">
{doc_text}
                        </div>
                        """),
                        unsafe_allow_html=True
                    )
    except Exception:
        st.error("Could not fetch document list.")