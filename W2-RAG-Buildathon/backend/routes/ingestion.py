"""Ingestion and index management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import Services, get_services
from backend.schemas import IndexStatus, IngestionSummary
from utils.logger import get_logger

router = APIRouter(tags=["Ingestion & Index"])
logger = get_logger(__name__)


def _run(svc: Services, rebuild: bool) -> IngestionSummary:
    try:
        return IngestionSummary(**svc.rag.ingest(rebuild=rebuild))
    except RuntimeError as exc:  # concurrent ingestion
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        svc.faiss.record_status("failed", str(exc))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Ingestion failed: {exc}") from exc


@router.post("/ingest", response_model=IngestionSummary,
             summary="Incrementally index uploaded PDFs that are not yet indexed")
def ingest(svc: Services = Depends(get_services)) -> IngestionSummary:
    return _run(svc, rebuild=False)


@router.post("/index/rebuild", response_model=IngestionSummary,
             summary="Rebuild the complete FAISS index from all stored PDFs")
def rebuild_index(svc: Services = Depends(get_services)) -> IngestionSummary:
    return _run(svc, rebuild=True)


@router.get("/index/status", response_model=IndexStatus, summary="FAISS index status")
def index_status(svc: Services = Depends(get_services)) -> IndexStatus:
    return IndexStatus(**svc.faiss.status())
