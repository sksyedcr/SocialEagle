"""Document upload / list / delete endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.dependencies import Services, get_services
from backend.schemas import DeleteResponse, DocumentListResponse, UploadResponse, UploadResult

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILES_PER_REQUEST = 20


@router.post("/upload", response_model=UploadResponse, summary="Upload one or more PDF files")
async def upload_documents(
    files: list[UploadFile] = File(..., description="PDF files to upload"),
    svc: Services = Depends(get_services),
) -> UploadResponse:
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files provided.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Maximum {MAX_FILES_PER_REQUEST} files per upload.")

    max_bytes = svc.settings.max_upload_mb * 1024 * 1024
    results: list[UploadResult] = []
    for f in files:
        data = await f.read(max_bytes + 1)  # bounded read: never buffer more than the limit
        await f.close()
        results.append(UploadResult(**svc.documents.save_upload(f.filename or "document.pdf", data)))
    saved = sum(1 for r in results if r.status == "Uploaded")
    return UploadResponse(saved=saved, results=results)


@router.get("", response_model=DocumentListResponse, summary="List uploaded documents and their status")
def list_documents(svc: Services = Depends(get_services)) -> DocumentListResponse:
    svc.documents.reconcile()
    docs = svc.documents.list_documents()
    return DocumentListResponse(count=len(docs), documents=docs)


@router.delete("/{document_id}", response_model=DeleteResponse, summary="Delete a PDF and its indexed chunks")
def delete_document(document_id: str, svc: Services = Depends(get_services)) -> DeleteResponse:
    if not document_id.isalnum() or len(document_id) > 64:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid document id.")
    result = svc.rag.delete_document(document_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found.")
    return DeleteResponse(**result, message=f"Deleted '{result['filename']}' and removed its indexed content.")
