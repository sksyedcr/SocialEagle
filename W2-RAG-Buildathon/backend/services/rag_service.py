"""RAG orchestration: ingestion workflow + question answering with watsonx.ai GPT-OSS-120B."""

from __future__ import annotations

import re
import threading
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from backend.services.document_service import DocumentService
from backend.services.faiss_service import FaissService
from config.settings import ConfigurationError, Settings
from pipelines.ingestion_pipeline import PDFLoadError, process_pdf
from pipelines.retrieval_pipeline import (
    ANSWER_PROMPT,
    CONDENSE_PROMPT,
    NO_CONTEXT_ANSWER,
    build_sources,
    format_context,
    format_history,
    retrieve,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """The LLM provider call failed."""


def build_llm(settings: Settings) -> BaseChatModel:
    """IBM watsonx.ai MaaS exposes an OpenAI-compatible API, so we use ChatOpenAI.

    No fallback provider: if credentials are missing we raise.
    """
    if not settings.watsonx_token:
        raise ConfigurationError("WATSONX_TOKEN is not set; cannot call watsonx.ai.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.watsonx_token,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=2,
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_llm_text(text: str) -> str:
    """Strip any leaked reasoning blocks some reasoning models emit."""
    return _THINK_RE.sub("", text or "").strip()


class RAGService:
    def __init__(
        self,
        settings: Settings,
        documents: DocumentService,
        faiss: FaissService,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.documents = documents
        self.faiss = faiss
        self._llm = llm
        self._ingest_lock = threading.Lock()

    # ------------------------------------------------------------------ LLM
    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # ------------------------------------------------------------------ ingestion
    def _process(self, doc_id: str) -> tuple[Any, str | None]:
        doc = self.documents.get(doc_id)
        assert doc is not None
        try:
            chunked = process_pdf(
                self.documents.path_for(doc_id),
                doc_id,
                doc["filename"],
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            return chunked, None
        except PDFLoadError as exc:
            return None, str(exc)
        except Exception as exc:  # unexpected parser failure: isolate to this document
            logger.exception("Failed to process %s", doc["filename"])
            return None, f"Unexpected error: {exc}"

    def ingest(self, rebuild: bool = False) -> dict[str, Any]:
        """Incrementally index pending documents, or rebuild everything.

        Incremental mode only embeds documents that are not yet Indexed, so
        existing vectors are never regenerated.
        """
        if not self._ingest_lock.acquire(blocking=False):
            raise RuntimeError("An ingestion is already running. Please wait for it to finish.")
        try:
            self.documents.reconcile()
            # Drop index content for documents no longer present on disk.
            for stale in self.faiss.indexed_document_ids() - set(self.documents.all_ids()):
                self.faiss.delete_document(stale)

            if rebuild:
                self.documents.mark_all_uploaded()
                target_ids = self.documents.all_ids()
            else:
                # New uploads, plus docs marked Indexed whose vectors are missing (e.g. index
                # deleted). Previously Failed docs are NOT retried here (they'd fail again
                # deterministically); "Rebuild Index" retries everything.
                indexed = self.faiss.indexed_document_ids()
                target_ids = [
                    d["document_id"] for d in self.documents.list_documents()
                    if d["status"] == "Uploaded" or (d["status"] == "Indexed" and d["document_id"] not in indexed)
                ]

            results: list[dict[str, Any]] = []
            batch: list[tuple[str, str, str, list, list]] = []
            pages_total = chunks_total = 0

            for doc_id in target_ids:
                doc = self.documents.get(doc_id)
                chunked, error = self._process(doc_id)
                if error:
                    self.documents.mark_failed(doc_id, error)
                    results.append({"document_id": doc_id, "filename": doc["filename"], "status": "Failed",
                                    "pages": None, "chunks": 0, "error": error})
                    continue
                pages_total += chunked.total_pages
                chunks_total += len(chunked.chunks)
                if rebuild:
                    batch.append((doc_id, doc["filename"], doc["checksum"], chunked.chunks, chunked.chunk_ids))
                else:
                    try:
                        self.faiss.add_documents(doc_id, doc["filename"], doc["checksum"],
                                                 chunked.chunks, chunked.chunk_ids)
                    except Exception as exc:
                        logger.exception("Indexing failed for %s", doc["filename"])
                        self.documents.mark_failed(doc_id, f"Indexing error: {exc}", chunked.total_pages)
                        results.append({"document_id": doc_id, "filename": doc["filename"], "status": "Failed",
                                        "pages": chunked.total_pages, "chunks": 0, "error": str(exc)})
                        continue
                    self.documents.mark_indexed(doc_id, chunked.total_pages, len(chunked.chunks))
                results.append({"document_id": doc_id, "filename": doc["filename"], "status": "Indexed",
                                "pages": chunked.total_pages, "chunks": len(chunked.chunks),
                                "empty_pages": len(chunked.empty_pages), "error": None})

            if rebuild:
                self.faiss.rebuild(batch)
                for doc_id, _f, _c, chunks, _ids in batch:
                    pages = next(r["pages"] for r in results if r["document_id"] == doc_id)
                    self.documents.mark_indexed(doc_id, pages, len(chunks))

            failed = sum(1 for r in results if r["status"] == "Failed")
            if not target_ids:
                status, message = "up_to_date", "All documents are already indexed. Nothing to do."
            elif failed == len(results):
                status, message = "failed", "No documents could be indexed."
            elif failed:
                status, message = "partial", f"{len(results) - failed} indexed, {failed} failed."
            else:
                status, message = "success", f"{len(results)} document(s) indexed."

            fstatus = self.faiss.status()
            summary = {
                "mode": "rebuild" if rebuild else "incremental",
                "status": status,
                "message": message,
                "uploaded_pdfs": len(self.documents.all_ids()),
                "processed_pdfs": len(results),
                "pages_processed": pages_total,
                "chunks_created": chunks_total,
                "vectors_indexed": sum(r["chunks"] for r in results if r["status"] == "Indexed"),
                "total_vectors_in_index": fstatus["vector_count"],
                "index_location": fstatus["index_path"],
                "documents": results,
            }
            self.faiss.record_status(status, message)
            logger.info("Ingestion summary: %s", {k: v for k, v in summary.items() if k != "documents"})
            return summary
        finally:
            self._ingest_lock.release()

    def delete_document(self, doc_id: str) -> dict[str, Any] | None:
        removed_chunks = self.faiss.delete_document(doc_id)
        doc = self.documents.delete(doc_id)
        if doc is None:
            return None
        self.faiss.record_status("success", f"Deleted '{doc['filename']}' ({removed_chunks} chunks removed).")
        return {"document_id": doc_id, "filename": doc["filename"], "chunks_removed": removed_chunks}

    # ------------------------------------------------------------------ chat
    def _condense(self, question: str, history: list[dict[str, str]]) -> str:
        """Turn a follow-up into a standalone retrieval query using only recent turns."""
        if not history:
            return question
        try:
            chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
            rewritten = _clean_llm_text(chain.invoke({
                "history": format_history(history, self.settings.history_turns, max_chars=300),
                "question": question,
            }))
            rewritten = rewritten.strip().strip('"').splitlines()[0] if rewritten else ""
            if 3 <= len(rewritten) <= 500:
                logger.info("Condensed follow-up %r -> %r", question, rewritten)
                return rewritten
        except Exception:
            logger.warning("Question condensation failed; using original question", exc_info=True)
        return question

    def answer(self, question: str, history: list[dict[str, str]] | None = None,
               top_k: int | None = None) -> dict[str, Any]:
        history = history or []
        k = top_k or self.settings.top_k
        if not self.faiss.is_ready:
            return {"answer": "No documents have been indexed yet. Upload PDFs in the left panel and click "
                              "'Process Documents' first.", "sources": [], "standalone_question": question}

        standalone = self._condense(question, history)
        chunks = retrieve(self.faiss, standalone, k=k)
        if not chunks:
            return {"answer": NO_CONTEXT_ANSWER, "sources": [], "standalone_question": standalone}

        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({
                "history": format_history(history, self.settings.history_turns),
                "context": format_context(chunks),
                "question": question if standalone == question else f"{question}\n(Interpreted as: {standalone})",
            })
        except Exception as exc:
            logger.error("LLM call failed: %s", type(exc).__name__)
            raise LLMError(f"watsonx.ai request failed: {type(exc).__name__}: {exc}") from exc

        return {
            "answer": _clean_llm_text(raw) or NO_CONTEXT_ANSWER,
            "sources": build_sources(chunks),
            "standalone_question": standalone,
        }
