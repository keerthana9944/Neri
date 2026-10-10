import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

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

    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='documents'"
    ).fetchone()

    if table_row is None:
        connection.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
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
    else:
        table_sql = table_row[0] or ""
        # Check if table lacks doc_id or still enforces filename UNIQUE
        if "doc_id" not in table_sql or "filename TEXT NOT NULL UNIQUE" in table_sql:
            from src.machine_norm import generate_document_id
            connection.execute("ALTER TABLE documents RENAME TO documents_old")
            connection.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
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
            old_rows = connection.execute("SELECT * FROM documents_old").fetchall()
            for row in old_rows:
                m = dict(row)
                calc_doc_id = m.get("doc_id") or generate_document_id(
                    document_type=m.get("document_type", "manual"),
                    machine=m.get("machine"),
                    filename=m.get("filename", "doc"),
                )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        id, doc_id, filename, document_type, machine, version, owner, chunks, status, uploaded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.get("id"),
                        calc_doc_id,
                        m.get("filename"),
                        m.get("document_type"),
                        m.get("machine"),
                        m.get("version"),
                        m.get("owner"),
                        m.get("chunks", 0),
                        m.get("status", "indexed"),
                        m.get("uploaded_at"),
                    ),
                )
            connection.execute("DROP TABLE documents_old")

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
    doc_id: str | None = None,
) -> int:
    """Save uploaded document metadata with unique doc_id."""

    initialize_database()

    from src.machine_norm import (
        generate_document_id,
        normalize_machine_id,
    )

    canonical_machine = normalize_machine_id(machine)

    if not doc_id:
        doc_id = generate_document_id(
            document_type=document_type,
            machine=canonical_machine,
            filename=filename,
        )

    connection = _get_connection()

    cursor = connection.execute(
        """
        INSERT OR REPLACE INTO documents (
            doc_id,
            filename,
            document_type,
            machine,
            version,
            owner,
            chunks,
            status,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            filename,
            document_type,
            canonical_machine,
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
            doc_id,
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


def delete_document(
    doc_id: str | None = None,
    filename: str | None = None,
    machine: str | None = None,
) -> bool:
    """
    Delete a document record from SQLite and purge its chunks from ChromaDB.
    """
    initialize_database()
    connection = _get_connection()
    deleted = False

    try:
        from src.vector_store import delete_document_chunks

        target_doc_id = doc_id
        target_filename = filename

        if target_doc_id:
            cursor = connection.execute(
                "DELETE FROM documents WHERE doc_id = ?",
                (target_doc_id,),
            )
            connection.commit()
            if cursor.rowcount > 0:
                deleted = True

            # Strictly delete only chunks matching target_doc_id
            delete_document_chunks(doc_id=target_doc_id)

        elif target_filename:
            from src.machine_norm import normalize_machine_id

            if machine:
                norm_m = normalize_machine_id(machine)
                rows = connection.execute(
                    "SELECT doc_id FROM documents WHERE filename = ? AND (machine = ? OR machine = ?)",
                    (target_filename, machine, norm_m),
                ).fetchall()
                for r in rows:
                    if r["doc_id"]:
                        delete_document_chunks(doc_id=r["doc_id"])

                cursor = connection.execute(
                    "DELETE FROM documents WHERE filename = ? AND (machine = ? OR machine = ?)",
                    (target_filename, machine, norm_m),
                )
                connection.commit()
                if cursor.rowcount > 0:
                    deleted = True

                delete_document_chunks(document_name=target_filename, machine=machine)
            else:
                rows = connection.execute(
                    "SELECT doc_id, machine FROM documents WHERE filename = ?",
                    (target_filename,),
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError(
                        f"Ambiguous document deletion: Multiple documents share filename '{target_filename}' across machines. Please specify doc_id or machine to delete."
                    )
                elif len(rows) == 1:
                    single_doc_id = rows[0]["doc_id"]
                    cursor = connection.execute(
                        "DELETE FROM documents WHERE doc_id = ?",
                        (single_doc_id,),
                    )
                    connection.commit()
                    if cursor.rowcount > 0:
                        deleted = True
                    delete_document_chunks(doc_id=single_doc_id)
                else:
                    return False

    finally:
        connection.close()

    return deleted


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

    uploads_dir = BASE_DIR / "uploads"
    if uploads_dir.exists():
        for file_path in uploads_dir.iterdir():
            if file_path.is_file() and file_path.name not in indexed_filenames:
                if file_path.suffix.lower() in [".pdf", ".txt"]:
                    needed_files.append((file_path, "maintenance_log"))

    if not needed_files:
        return

    print(
        f"Auto-seeding {len(needed_files)} missing data files into knowledge base..."
    )

    from src.ingestion import create_chunks
    from src.machine_norm import (
        generate_document_id,
        normalize_machine_id,
    )
    from src.vector_store import add_documents

    for file_path, doc_type in needed_files:
        base_stem = file_path.stem
        for suffix in ["_manual", "_maintenance", "_safety"]:
            if base_stem.endswith(suffix):
                base_stem = base_stem[:-len(suffix)]
        machine_name = " ".join(
            [w.capitalize() for w in base_stem.split("_")]
        )
        canonical_machine = normalize_machine_id(machine_name)
        doc_id = generate_document_id(
            document_type=doc_type,
            machine=canonical_machine,
            filename=file_path.name,
        )

        chunks = create_chunks(
            file_path=file_path,
            document_type=doc_type,
            machine=canonical_machine,
            version="1.0",
            owner="Maintenance Department",
            doc_id=doc_id,
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
            machine=canonical_machine,
            version="1.0",
            owner="Maintenance Department",
            chunks=len(chunks),
            status="indexed",
            doc_id=doc_id,
        )


def _sort_chunks(metadatas: list[dict], documents: list[str]) -> list[str]:
    def sort_key(item):
        meta, _ = item
        chunk_id = str(meta.get("chunk_id", ""))
        page = meta.get("page") or 0
        try:
            num = int(chunk_id.rsplit("_c", 1)[-1]) if "_c" in chunk_id else int(chunk_id.rsplit("_", 1)[-1])
        except Exception:
            num = 0
        return (page, num, chunk_id)

    paired = list(zip(metadatas, documents))
    paired.sort(key=sort_key)
    return [doc for _, doc in paired]


def normalize_document_text(text: str) -> str:
    """
    Normalizes extracted document text for preview and display:
    - Normalizes repeated spaces and blank lines while preserving paragraph boundaries,
      headings, lists, and meaningful document structure.
    - Resolves word-per-line extraction artifacts from PDFs.
    - Strips trailing whitespace per line.
    """
    if not text or not str(text).strip():
        return ""

    raw = str(text).replace("\ufffd", "—")

    # Detect word-per-line artifact (where lines are mostly isolated single words)
    raw_lines = raw.splitlines()
    non_empty = [l.strip() for l in raw_lines if l.strip()]
    if len(non_empty) > 15:
        avg_words = sum(len(l.split()) for l in non_empty) / len(non_empty)
        if avg_words <= 1.3:
            raw = " ".join(non_empty)
            raw = re.sub(
                r'(\b(?:Machine Information|Document Information|Observation|Action Taken|Result|Follow-up|Escalation Guidance|Maintenance Record \d+|Problem:.*)\b)',
                r'\n\n\1\n',
                raw,
            )

    # Normalize horizontal spacing and trailing whitespace on each line
    cleaned_lines = []
    for line in raw.splitlines():
        trimmed = line.strip()
        if not trimmed:
            cleaned_lines.append("")
        else:
            collapsed = re.sub(r"[ \t]{2,}", " ", trimmed)
            cleaned_lines.append(collapsed)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def get_document_content(
    identifier: str,
    machine: str | None = None,
) -> str:
    """
    Retrieve full text content of a document by doc_id or filename.
    If multiple documents share the same filename across machines,
    requires doc_id or machine to prevent ambiguous returns.
    """
    initialize_database()
    connection = _get_connection()
    matched_doc_id = None
    target_filename = identifier

    # 1. Check if identifier is a doc_id
    row = connection.execute(
        "SELECT doc_id, filename, machine FROM documents WHERE doc_id = ?",
        (identifier,),
    ).fetchone()

    if row:
        matched_doc_id = row["doc_id"]
        target_filename = row["filename"]
    else:
        # 2. Check if identifier matches filename in documents table
        rows = connection.execute(
            "SELECT doc_id, filename, machine FROM documents WHERE filename = ?",
            (identifier,),
        ).fetchall()

        if len(rows) > 1:
            if machine:
                from src.machine_norm import normalize_machine_id
                norm_m = normalize_machine_id(machine)
                filtered = [r for r in rows if r["machine"] == norm_m or r["machine"] == machine]
                if len(filtered) == 1:
                    matched_doc_id = filtered[0]["doc_id"]
                    target_filename = filtered[0]["filename"]
                else:
                    connection.close()
                    machines_list = ", ".join(r["machine"] for r in rows)
                    return f"Ambiguous document lookup: Multiple documents share filename '{identifier}' across machines ({machines_list}). Please specify a doc_id."
            else:
                connection.close()
                machines_list = ", ".join(r["machine"] for r in rows)
                return f"Ambiguous document lookup: Multiple documents share filename '{identifier}' across machines ({machines_list}). Please specify a doc_id."
        elif len(rows) == 1:
            matched_doc_id = rows[0]["doc_id"]
            target_filename = rows[0]["filename"]

    connection.close()

    # 3. Retrieve from ChromaDB if matched_doc_id is known (avoids file-overwrite issues)
    if matched_doc_id:
        try:
            from src.vector_store import get_collection
            collection = get_collection()
            res = collection.get(where={"doc_id": matched_doc_id})
            if res and res.get("documents") and len(res["documents"]) > 0:
                docs = res["documents"]
                metadatas = res.get("metadatas") or []
                if len(metadatas) == len(docs):
                    docs = _sort_chunks(metadatas, docs)
                return normalize_document_text("\n\n".join(docs))
        except Exception:
            pass

    # 4. Check on-disk repositories for data/ and uploads/
    fname = Path(target_filename).name

    for sub in ["manuals", "maintenance_logs", "safety"]:
        path = BASE_DIR / "data" / sub / fname
        if path.exists():
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    return normalize_document_text(txt)
            except Exception:
                pass

    up_path = BASE_DIR / "uploads" / fname
    if up_path.exists():
        if up_path.suffix.lower() == ".txt":
            try:
                txt = up_path.read_text(encoding="utf-8", errors="ignore").strip()
                if txt:
                    return normalize_document_text(txt)
            except Exception:
                pass
        elif up_path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(up_path))
                pages = []
                for page in reader.pages:
                    try:
                        txt = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
                    except Exception:
                        txt = page.extract_text() or ""
                    if txt and txt.strip():
                        pages.append(txt.strip())
                if pages:
                    return normalize_document_text("\n\n".join(pages))
            except Exception:
                pass

    # 5. Fallback to ChromaDB by filename and machine
    try:
        from src.vector_store import get_collection
        collection = get_collection()
        if machine:
            from src.machine_norm import get_machine_retrieval_aliases
            aliases = get_machine_retrieval_aliases(machine)
            if aliases:
                res = collection.get(
                    where={
                        "$and": [
                            {"document": {"$eq": fname}},
                            {"machine": {"$in": aliases}},
                        ]
                    }
                )
            else:
                res = collection.get(where={"document": fname})
        else:
            res = collection.get(where={"document": fname})

        if res and res.get("documents") and len(res["documents"]) > 0:
            docs = res["documents"]
            metadatas = res.get("metadatas") or []
            if len(metadatas) == len(docs):
                docs = _sort_chunks(metadatas, docs)
            return normalize_document_text("\n\n".join(docs))
    except Exception:
        pass

    return "Document content unavailable."



