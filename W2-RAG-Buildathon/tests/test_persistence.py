"""Stage 6/7: FAISS creation, persistence, reload without re-embedding, incremental add, delete, rebuild."""

from __future__ import annotations

from dataclasses import replace

from backend.services.document_service import DocumentService
from backend.services.faiss_service import FaissService
from backend.services.rag_service import RAGService
from tests.conftest import HashEmbeddings, make_pdf


class CountingEmbeddings(HashEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.doc_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += len(texts)
        return super().embed_documents(texts)


def _rag(settings, emb, llm=None):
    docs = DocumentService(settings)
    faiss = FaissService(settings, embeddings=emb)
    faiss.load()
    return RAGService(settings, docs, faiss, llm=llm), docs, faiss


def test_first_run_creates_and_persists_index(settings, embeddings, pdf_a):
    rag, docs, faiss = _rag(settings, embeddings)
    docs.save_upload("scheme_a.pdf", pdf_a)
    summary = rag.ingest()
    assert summary["status"] == "success"
    assert summary["uploaded_pdfs"] == 1
    assert summary["pages_processed"] == 3
    assert summary["chunks_created"] == summary["vectors_indexed"] > 0
    assert (settings.faiss_index_dir / "index.faiss").exists()
    assert (settings.faiss_index_dir / "index_manifest.json").exists()
    assert docs.list_documents()[0]["status"] == "Indexed"
    assert docs.list_documents()[0]["pages"] == 3


def test_reload_after_restart_without_reembedding(settings, pdf_a):
    rag, docs, _ = _rag(settings, CountingEmbeddings())
    docs.save_upload("scheme_a.pdf", pdf_a)
    n = rag.ingest()["vectors_indexed"]

    # Simulated restart: brand-new objects reading from disk.
    emb2 = CountingEmbeddings()
    rag2, docs2, faiss2 = _rag(settings, emb2)
    assert faiss2.is_ready and faiss2.status()["vector_count"] == n
    assert emb2.doc_calls == 0  # nothing re-embedded on load
    assert rag2.ingest()["status"] == "up_to_date"
    assert emb2.doc_calls == 0  # nothing re-embedded on a no-op ingest
    assert faiss2.search("eligibility small farmers", k=1)[0][0].metadata["source"] == "scheme_a.pdf"


def test_incremental_add_only_embeds_new_document(settings, pdf_a, pdf_b):
    emb = CountingEmbeddings()
    rag, docs, faiss = _rag(settings, emb)
    docs.save_upload("scheme_a.pdf", pdf_a)
    first = rag.ingest()["vectors_indexed"]
    calls_after_first = emb.doc_calls

    docs.save_upload("scheme_b.pdf", pdf_b)
    second = rag.ingest()
    assert second["processed_pdfs"] == 1
    assert emb.doc_calls - calls_after_first == second["vectors_indexed"]
    assert faiss.status()["vector_count"] == first + second["vectors_indexed"]
    assert faiss.status()["indexed_documents"] == 2


def test_delete_removes_chunks_from_index_and_disk(settings, embeddings, pdf_a, pdf_b):
    rag, docs, faiss = _rag(settings, embeddings)
    a = docs.save_upload("scheme_a.pdf", pdf_a)["document_id"]
    docs.save_upload("scheme_b.pdf", pdf_b)
    rag.ingest()

    res = rag.delete_document(a)
    assert res["chunks_removed"] > 0
    assert not (settings.documents_dir / "scheme_a.pdf").exists()
    hits = faiss.search("eligibility small marginal farmers Aadhaar patta", k=10)
    assert hits and all(d.metadata["source"] != "scheme_a.pdf" for d, _ in hits)

    # Deletion is persisted: a restart must not resurrect deleted chunks.
    _, _, faiss2 = _rag(settings, embeddings)
    assert all(d.metadata["source"] != "scheme_a.pdf"
               for d, _ in faiss2.search("eligibility small farmers", k=10))


def test_deleting_last_document_empties_index(settings, embeddings, pdf_a):
    rag, docs, faiss = _rag(settings, embeddings)
    a = docs.save_upload("scheme_a.pdf", pdf_a)["document_id"]
    rag.ingest()
    rag.delete_document(a)
    assert not faiss.is_ready
    assert not (settings.faiss_index_dir / "index.faiss").exists()


def test_rebuild_replaces_index(settings, embeddings, pdf_a, pdf_b):
    rag, docs, faiss = _rag(settings, embeddings)
    docs.save_upload("scheme_a.pdf", pdf_a)
    docs.save_upload("scheme_b.pdf", pdf_b)
    first = rag.ingest()
    summary = rag.ingest(rebuild=True)
    assert summary["mode"] == "rebuild"
    assert summary["processed_pdfs"] == 2
    assert faiss.status()["vector_count"] == first["vectors_indexed"]  # no duplicates
    assert all(d["status"] == "Indexed" for d in docs.list_documents())


def test_reingest_same_doc_no_duplicate_vectors(settings, embeddings, pdf_a):
    rag, docs, faiss = _rag(settings, embeddings)
    doc_id = docs.save_upload("scheme_a.pdf", pdf_a)["document_id"]
    n = rag.ingest()["vectors_indexed"]
    # Directly re-add the same document: its old chunks must be replaced, not duplicated.
    from pipelines.ingestion_pipeline import process_pdf
    c = process_pdf(docs.path_for(doc_id), doc_id, "scheme_a.pdf", settings.chunk_size, settings.chunk_overlap)
    faiss.add_documents(doc_id, "scheme_a.pdf", "x", c.chunks, c.chunk_ids)
    assert faiss.status()["vector_count"] == n


def test_failed_pdf_isolated_and_not_retried_incrementally(settings, embeddings, pdf_a, pdf_b):
    rag, docs, faiss = _rag(settings, embeddings)
    docs.save_upload("scheme_a.pdf", pdf_a)
    docs.save_upload("scanned.pdf", make_pdf(["", ""]))
    summary = rag.ingest()
    assert summary["status"] == "partial"
    statuses = {d["filename"]: d["status"] for d in docs.list_documents()}
    assert statuses == {"scheme_a.pdf": "Indexed", "scanned.pdf": "Failed"}
    assert faiss.is_ready

    # A later upload only processes the new file; the known-bad PDF is not re-parsed.
    docs.save_upload("scheme_b.pdf", pdf_b)
    summary = rag.ingest()
    assert summary["status"] == "success"
    assert [d["filename"] for d in summary["documents"]] == ["scheme_b.pdf"]
    assert rag.ingest()["status"] == "up_to_date"

    # Rebuild retries everything, including the failed one.
    summary = rag.ingest(rebuild=True)
    assert summary["processed_pdfs"] == 3 and summary["status"] == "partial"


def test_incompatible_embedding_model_not_loaded(settings, embeddings, pdf_a):
    rag, docs, _ = _rag(settings, embeddings)
    docs.save_upload("scheme_a.pdf", pdf_a)
    rag.ingest()
    other = FaissService(replace(settings, embedding_model="some/other-model"), embeddings=embeddings)
    assert other.load() is False
    assert "Rebuild" in other.load_error


def test_untrusted_index_without_manifest_not_loaded(settings, embeddings, pdf_a):
    rag, docs, _ = _rag(settings, embeddings)
    docs.save_upload("scheme_a.pdf", pdf_a)
    rag.ingest()
    (settings.faiss_index_dir / "index_manifest.json").unlink()
    fresh = FaissService(settings, embeddings=embeddings)
    assert fresh.load() is False
    assert not fresh.is_ready
