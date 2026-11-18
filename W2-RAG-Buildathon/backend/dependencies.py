"""Application service container (created once at startup, injected into routes)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from backend.services.document_service import DocumentService
from backend.services.faiss_service import FaissService
from backend.services.rag_service import RAGService
from config.settings import Settings


@dataclass
class Services:
    settings: Settings
    documents: DocumentService
    faiss: FaissService
    rag: RAGService


def build_services(settings: Settings, faiss: FaissService | None = None, llm=None) -> Services:
    documents = DocumentService(settings)
    faiss = faiss or FaissService(settings)
    rag = RAGService(settings, documents, faiss, llm=llm)
    return Services(settings, documents, faiss, rag)


def get_services(request: Request) -> Services:
    return request.app.state.services
