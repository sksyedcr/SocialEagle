"""FastAPI entrypoint.  Run with:  uvicorn backend.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from backend.dependencies import Services, build_services, get_services
from backend.routes import chat, documents, ingestion
from backend.schemas import HealthResponse
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("backend")


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = services or build_services(get_settings())
        problems = svc.settings.validate()
        if problems:
            for p in problems:
                logger.error("CONFIG: %s", p)
            logger.error("Chat will be unavailable until configuration is fixed (upload/indexing still work).")
        else:
            logger.info("LLM configured: model=%s base_url=%s (token loaded, not shown)",
                        svc.settings.llm_model, svc.settings.llm_base_url)
        # Load the persisted index (no re-embedding).
        try:
            svc.faiss.load()
        except Exception:
            logger.exception("Failed to load FAISS index; use 'Rebuild Index' to recreate it")
            svc.faiss.load_error = "Index could not be loaded; rebuild required."
        app.state.services = svc
        yield

    app = FastAPI(
        title="Tamil Nadu Agriculture Schemes RAG API",
        description="Upload scheme PDFs, index them into a persistent FAISS store and ask grounded questions "
                    "answered by IBM watsonx.ai GPT-OSS-120B.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health(svc: Services = Depends(get_services)) -> HealthResponse:
        st = svc.faiss.status()
        problems = svc.settings.validate()
        return HealthResponse(
            status="ok",
            index_available=st["available"],
            vector_count=st["vector_count"],
            llm_configured=svc.settings.llm_configured,
            config_problems=problems,
        )

    app.include_router(documents.router)
    app.include_router(ingestion.router)
    app.include_router(chat.router)
    return app


app = create_app()
