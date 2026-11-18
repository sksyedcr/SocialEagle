"""Chat endpoint: grounded answers with citations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import Services, get_services
from backend.schemas import ChatRequest, ChatResponse
from backend.services.rag_service import LLMError
from config.settings import ConfigurationError
from utils.logger import get_logger

router = APIRouter(tags=["Chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse, summary="Ask a question about the uploaded scheme PDFs")
def chat(req: ChatRequest, svc: Services = Depends(get_services)) -> ChatResponse:
    try:
        result = svc.rag.answer(req.question, [t.model_dump() for t in req.history], top_k=req.top_k)
    except ConfigurationError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return ChatResponse(**result)
