import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

try:
    from src.config import (
        UPLOAD_DIR,
        get_supervisor_api_key,
        get_technician_api_key,
    )
except ImportError:
    from config import (
        UPLOAD_DIR,
        get_supervisor_api_key,
        get_technician_api_key,
    )
from src.embeddings import create_embeddings
from src.history import (
    get_documents,
    get_history,
    get_session,
    save_document,
    save_feedback,
    save_session,
)
from src.ingestion import create_chunks
from src.pipeline import troubleshoot
from src.vector_store import add_documents


app = FastAPI(
    title="Neri API",
    description="AI-powered maintenance troubleshooting assistant",
    version="1.0.0",
)


def get_current_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> dict:
    """Validate API authentication token from request headers."""
    token = x_api_key
    if not token and authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if not token or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="Authentication credentials are required.",
        )

    supervisor_key = get_supervisor_api_key()
    technician_key = get_technician_api_key()

    # Fail closed if server keys are not configured properly or are identical
    if not supervisor_key or not technician_key or supervisor_key == technician_key:
        raise HTTPException(
            status_code=500,
            detail="Server authentication configuration is invalid or missing.",
        )

    if token == supervisor_key:
        return {"username": "supervisor", "role": "Supervisor"}
    elif token == technician_key:
        return {"username": "technician", "role": "Technician"}
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
        )


def require_supervisor(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Ensure current user has Supervisor role."""
    if current_user.get("role") != "Supervisor":
        raise HTTPException(
            status_code=403,
            detail="Supervisor privileges required for this action.",
        )
    return current_user


class TroubleshootingRequest(BaseModel):
    machine: str
    machine_id: str
    problem: str
    error_code: Optional[str] = None


@app.get("/")
def root():
    """Health check for the Neri API."""

    return {
        "name": "Neri",
        "status": "running",
    }


@app.get("/health")
def health():
    """API health check."""

    return {
        "status": "healthy",
    }


@app.post("/query")
def query(
    request: TroubleshootingRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run Neri troubleshooting and save the session."""

    try:
        response = troubleshoot(
            machine=request.machine,
            machine_id=request.machine_id,
            problem=request.problem,
            error_code=request.error_code,
        )

        session_id = save_session(
            machine=request.machine,
            machine_id=request.machine_id,
            problem=request.problem,
            error_code=request.error_code,
            response=response,
        )

        response["session_id"] = session_id

        return response

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Neri could not process the troubleshooting request.",
        ) from error


@app.get("/history")
def history(current_user: dict = Depends(get_current_user)):
    """Return recent troubleshooting sessions."""

    try:
        return {
            "sessions": get_history()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve troubleshooting history.",
        ) from error


@app.get("/history/{session_id}")
def history_detail(
    session_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Return one troubleshooting session."""

    try:
        session = get_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Troubleshooting session not found.",
            )

        return session

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve troubleshooting session.",
        ) from error


#---------------------------------#
# Feedback
#---------------------------------#
class FeedbackRequest(BaseModel):
    feedback: str
    comment: Optional[str] = None


@app.post("/history/{session_id}/feedback")
def feedback(
    session_id: int,
    request: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save feedback for a troubleshooting session."""

    try:
        updated = save_feedback(
            session_id=session_id,
            feedback=request.feedback,
            feedback_comment=request.comment,
        )

        if not updated:
            raise HTTPException(
                status_code=404,
                detail="Troubleshooting session not found.",
            )

        return {
            "success": True,
            "session_id": session_id,
            "feedback": request.feedback,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not save feedback.",
        ) from error


#---------------------------------#
# Document upload
#---------------------------------#

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    machine: Optional[str] = Form(None),
    version: Optional[str] = Form(None),
    owner: Optional[str] = Form(None),
    current_user: dict = Depends(require_supervisor),
):
    """Upload and index an approved Neri document (Supervisor only)."""

    allowed_types = {
        "manual",
        "maintenance_log",
        "safety",
    }

    if document_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid document type. "
                "Use manual, maintenance_log, or safety."
            ),
        )

    allowed_extensions = {
        ".pdf",
        ".txt",
    }

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    try:
        UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = UPLOAD_DIR / Path(
            file.filename
        ).name

        file_content = await file.read()

        file_path.write_bytes(
            file_content
        )

        from src.machine_norm import (
            generate_document_id,
            normalize_machine_id,
        )
        from src.vector_store import delete_document_chunks

        canonical_machine = normalize_machine_id(machine)
        doc_id = generate_document_id(
            document_type=document_type,
            machine=canonical_machine,
            filename=file.filename,
        )

        # Purge existing chunks for clean re-ingestion (prevents duplicates & orphans)
        delete_document_chunks(
            doc_id=doc_id,
            document_name=file.filename,
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
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the document.",
            )

        embeddings = create_embeddings(
            [
                chunk["text"]
                for chunk in chunks
            ]
        )

        add_documents(
            chunks=chunks,
            embeddings=embeddings,
            clean_existing=False,
        )

        save_document(
            filename=file.filename,
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
            "document": file.filename,
            "document_type": document_type,
            "machine": canonical_machine,
            "version": version,
            "owner": owner,
            "chunks": len(chunks),
            "status": "indexed",
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not process the document.",
        ) from error

#------------------------------#
# Retrieve document
#------------------------------#
@app.get("/documents")
def documents(current_user: dict = Depends(get_current_user)):
    """Return uploaded documents."""

    try:
        return {
            "documents": get_documents()
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve documents.",
        ) from error


@app.get("/documents/{filename}/content")
def document_content(
    filename: str,
    machine: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    """Return full content of a document by filename or doc_id."""

    try:
        from src.history import get_document_content
        return {
            "filename": filename,
            "content": get_document_content(filename, machine=machine)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve document content.",
        ) from error


#------------------------------#
# Delete document
#------------------------------#
@app.delete("/documents/{doc_id}")
def delete_document_endpoint(
    doc_id: str,
    current_user: dict = Depends(require_supervisor),
):
    """Delete an uploaded document and its chunks (Supervisor only)."""

    try:
        from src.history import delete_document
        deleted = delete_document(doc_id=doc_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{doc_id}' not found.",
            )

        return {
            "success": True,
            "message": f"Document '{doc_id}' deleted successfully.",
            "doc_id": doc_id,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not delete document.",
        ) from error
