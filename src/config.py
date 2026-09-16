import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"

MANUALS_DIR = DATA_DIR / "manuals"
MAINTENANCE_LOGS_DIR = DATA_DIR / "maintenance_logs"
SAFETY_DIR = DATA_DIR / "safety"

UPLOAD_DIR = BASE_DIR / "uploads"

# Gemini / OpenAI-compatible API
def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    return key


OPENAI_API_KEY = get_openai_api_key()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.1-flash-lite",
)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "gemini-embedding-001",
)


# Vector database
VECTOR_DIR = BASE_DIR / os.getenv(
    "VECTOR_DIR",
    "storage/chroma_db",
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag_chunks",
)


# API
RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)


# Cache
CACHE_TTL_SECONDS = int(
    os.getenv("CACHE_TTL_SECONDS", "900")
)


# Model cost estimation
MODEL_INPUT_COST_PER_1K = float(
    os.getenv("MODEL_INPUT_COST_PER_1K", "0.00015")
)

MODEL_OUTPUT_COST_PER_1K = float(
    os.getenv("MODEL_OUTPUT_COST_PER_1K", "0.00060")
)


def validate_config():
    """Validate required Neri configuration."""

    key = get_openai_api_key()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add it to your .env file or Streamlit secrets."
        )

    return True