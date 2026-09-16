import json
import sqlite3
from datetime import datetime

from src.config import BASE_DIR


DATABASE_PATH = BASE_DIR / "storage" / "neri_history.db"


def _get_connection():
    """Create a connection to the Neri history database."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """Create and update the Neri database tables."""

    connection = _get_connection()

    # --------------------------------------------------
    # Troubleshooting History Table
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS troubleshooting_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine TEXT NOT NULL,
            machine_id TEXT NOT NULL,
            problem TEXT NOT NULL,
            error_code TEXT,
            response TEXT NOT NULL,
            feedback TEXT,
            feedback_comment TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # Check whether this is an older database.
    columns = connection.execute(
        "PRAGMA table_info(troubleshooting_history)"
    ).fetchall()

    column_names = {
        column["name"]
        for column in columns
    }

    # Add feedback column if it does not exist.
    if "feedback" not in column_names:
        connection.execute(
            """
            ALTER TABLE troubleshooting_history
            ADD COLUMN feedback TEXT
            """
        )

    # Add feedback comment column if it does not exist.
    if "feedback_comment" not in column_names:
        connection.execute(
            """
            ALTER TABLE troubleshooting_history
            ADD COLUMN feedback_comment TEXT
            """
        )

    # --------------------------------------------------
    # Documents Table
    # --------------------------------------------------

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            document_type TEXT NOT NULL,
            machine TEXT,
            version TEXT,
            owner TEXT,
            chunks INTEGER NOT NULL,
            status TEXT NOT NULL,
            uploaded_at TEXT NOT NULL
        )
        """
    )

    # Check whether the documents table is an older version.
    document_columns = connection.execute(
        "PRAGMA table_info(documents)"
    ).fetchall()

    document_column_names = {
        column["name"]
        for column in document_columns
    }

    # Add machine column if it does not exist.
    if "machine" not in document_column_names:
        connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN machine TEXT
            """
        )

    # Add version column if it does not exist.
    if "version" not in document_column_names:
        connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN version TEXT
            """
        )

    # Add owner column if it does not exist.
    if "owner" not in document_column_names:
        connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN owner TEXT
            """
        )

    connection.commit()
    connection.close()


# ======================================================
# Troubleshooting History
# ======================================================


def save_session(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None,
    response: dict,
) -> int:
    """Save a troubleshooting session."""

    initialize_database()

    connection = _get_connection()

    cursor = connection.execute(
        """
        INSERT INTO troubleshooting_history (
            machine,
            machine_id,
            problem,
            error_code,
            response,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            machine,
            machine_id,
            problem,
            error_code,
            json.dumps(response),
            datetime.now().isoformat(),
        ),
    )

    connection.commit()

    session_id = cursor.lastrowid

    connection.close()

    return session_id


def get_history(
    limit: int = 50,
) -> list[dict]:
    """Return recent troubleshooting sessions."""

    initialize_database()

    connection = _get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            machine,
            machine_id,
            problem,
            error_code,
            feedback,
            feedback_comment,
            created_at
        FROM troubleshooting_history
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_session(
    session_id: int,
) -> dict | None:
    """Return one complete troubleshooting session."""

    initialize_database()

    connection = _get_connection()

    row = connection.execute(
        """
        SELECT *
        FROM troubleshooting_history
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()

    connection.close()

    if row is None:
        return None

    session = dict(row)

    session["response"] = json.loads(
        session["response"]
    )

    return session


def save_feedback(
    session_id: int,
    feedback: str,
    feedback_comment: str | None = None,
) -> bool:
    """Save feedback for a troubleshooting session."""

    if feedback not in {
        "helpful",
        "not_helpful",
    }:
        raise ValueError(
            "Feedback must be 'helpful' or 'not_helpful'."
        )

    initialize_database()

    connection = _get_connection()

    cursor = connection.execute(
        """
        UPDATE troubleshooting_history
        SET
            feedback = ?,
            feedback_comment = ?
        WHERE id = ?
        """,
        (
            feedback,
            feedback_comment,
            session_id,
        ),
    )

    connection.commit()

    updated = cursor.rowcount > 0

    connection.close()

    return updated


# ======================================================
# Document Management
# ======================================================


def save_document(
    filename: str,
    document_type: str,
    chunks: int,
    machine: str | None = None,
    version: str | None = None,
    owner: str | None = None,
    status: str = "indexed",
) -> int:
    """Save uploaded document metadata."""

    initialize_database()

    connection = _get_connection()

    cursor = connection.execute(
        """
        INSERT OR REPLACE INTO documents (
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            datetime.now().isoformat(),
        ),
    )

    connection.commit()

    document_id = cursor.lastrowid

    connection.close()

    return document_id


def get_documents(
    limit: int = 100,
) -> list[dict]:
    """Return uploaded document metadata."""

    initialize_database()

    connection = _get_connection()

    rows = connection.execute(
        """
        SELECT
            id,
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            uploaded_at
        FROM documents
        ORDER BY uploaded_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def auto_seed_if_empty():
    """
    Check if the documents table or knowledge base is empty.
    If empty or if files in data/ are not indexed, automatically index all files under data/.
    """
    initialize_database()
    existing_documents = get_documents(limit=1000)
    indexed_filenames = {
        doc["filename"]
        for doc in existing_documents
        if doc.get("status") == "indexed"
    }

    dirs = [
        (BASE_DIR / "data" / "manuals", "manual"),
        (BASE_DIR / "data" / "maintenance_logs", "maintenance_log"),
        (BASE_DIR / "data" / "safety", "safety"),
    ]

    needed_files = []
    for dir_path, doc_type in dirs:
        if dir_path.exists():
            for file_path in dir_path.glob("*.txt"):
                if file_path.name not in indexed_filenames:
                    needed_files.append((file_path, doc_type))

    if not needed_files:
        return

    print(
        f"Auto-seeding {len(needed_files)} missing data files into knowledge base..."
    )

    from src.ingestion import create_chunks
    from src.vector_store import add_documents

    for file_path, doc_type in needed_files:
        base_stem = file_path.stem
        for suffix in ["_manual", "_maintenance", "_safety"]:
            if base_stem.endswith(suffix):
                base_stem = base_stem[:-len(suffix)]
        machine_name = " ".join(
            [w.capitalize() for w in base_stem.split("_")]
        )

        chunks = create_chunks(
            file_path=file_path,
            document_type=doc_type,
            machine=machine_name,
            version="1.0",
            owner="Maintenance Department",
        )

        if not chunks:
            continue

        try:
            from src.embeddings import create_embeddings
            embeddings = create_embeddings(
                [c["text"] for c in chunks],
                batch_size=20,
            )
            add_documents(
                chunks=chunks,
                embeddings=embeddings,
            )
        except Exception as err:
            print(
                f"Warning: Embeddings creation skipped for {file_path.name}: {err}"
            )

        save_document(
            filename=file_path.name,
            document_type=doc_type,
            machine=machine_name,
            version="1.0",
            owner="Maintenance Department",
            chunks=len(chunks),
            status="indexed",
        )


def get_document_content(filename: str) -> str:
    """Retrieve full text content of a document by filename."""

    for sub in ["manuals", "maintenance_logs", "safety"]:
        path = BASE_DIR / "data" / sub / filename
        if path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

    up_path = BASE_DIR / "uploads" / filename
    if up_path.exists():
        if up_path.suffix.lower() == ".txt":
            try:
                return up_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
        elif up_path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(up_path))
                pages = [
                    page.extract_text()
                    for page in reader.pages
                    if page.extract_text()
                ]
                return "\n\n".join(pages)
            except Exception:
                pass

    try:
        from src.vector_store import get_collection
        collection = get_collection()
        res = collection.get(where={"document": filename})
        if res and res.get("documents"):
            return "\n\n--- Chunk Divider ---\n\n".join(res["documents"])
    except Exception:
        pass

    return "Document content unavailable."
