from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.history import (
    get_documents,
    get_history,
    get_session,
    save_document,
    save_feedback,
    save_session,
)
from src.pipeline import troubleshoot

from src.config import UPLOAD_DIR
from src.embeddings import create_embeddings
from src.ingestion import create_chunks
from src.vector_store import add_documents


app = FastAPI(
    title="Neri API",
    description="AI-powered maintenance troubleshooting assistant",
    version="1.0.0",
)


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
def query(request: TroubleshootingRequest):
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
def history():
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
def history_detail(session_id: int):
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
):
    """Upload and index an approved Neri document."""

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

        chunks = create_chunks(
            file_path,
            document_type=document_type,
            machine=machine,
            version=version,
            owner=owner,
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
        )

        save_document(
            filename=file.filename,
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
            "document": file.filename,
            "document_type": document_type,
            "machine": machine,
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
def documents():
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
def document_content(filename: str):
    """Return full content of a document."""

    try:
        from src.history import get_document_content
        return {
            "filename": filename,
            "content": get_document_content(filename)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve document content.",
        ) from error