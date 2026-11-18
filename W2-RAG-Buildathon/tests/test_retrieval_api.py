"""Stage 8/10/11: retrieval, citations, prompt grounding and the full FastAPI surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.dependencies import build_services
from backend.main import create_app
from backend.services.faiss_service import FaissService
from pipelines.retrieval_pipeline import SYSTEM_PROMPT, build_sources, format_history, retrieve


@pytest.fixture()
def client(settings, embeddings, fake_llm):
    svc = build_services(settings, faiss=FaissService(settings, embeddings=embeddings), llm=fake_llm)
    with TestClient(create_app(svc)) as c:
        c.svc = svc
        yield c


def _upload(client, name, data):
    return client.post("/documents/upload", files=[("files", (name, data, "application/pdf"))])


def test_health_and_empty_index(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["index_available"] is False
    r = client.post("/chat", json={"question": "What schemes exist?"}).json()
    assert "No documents have been indexed" in r["answer"] and r["sources"] == []


def test_full_flow_upload_ingest_chat_delete(client, pdf_a, pdf_b):
    up = _upload(client, "scheme_a.pdf", pdf_a).json()
    assert up["saved"] == 1 and up["results"][0]["status"] == "Uploaded"
    _upload(client, "scheme_b.pdf", pdf_b)

    docs = client.get("/documents").json()
    assert docs["count"] == 2 and {d["status"] for d in docs["documents"]} == {"Uploaded"}

    summary = client.post("/ingest").json()
    assert summary["status"] == "success" and summary["processed_pdfs"] == 2
    assert {"pages_processed", "chunks_created", "vectors_indexed", "index_location"} <= summary.keys()

    status = client.get("/index/status").json()
    assert status["available"] and status["vector_count"] == summary["vectors_indexed"]
    assert status["embedding_model"] == "test/hash-embeddings"

    chat = client.post("/chat", json={"question": "What is the eligibility for small marginal farmers?"}).json()
    assert chat["answer"].startswith("Grounded answer")
    top = chat["sources"][0]
    assert top["document"] == "scheme_a.pdf" and top["page"] == 2
    assert "Eligibility" in top["relevant_content"]

    chat = client.post("/chat", json={"question": "drip irrigation subsidy"}).json()
    assert chat["sources"][0]["document"] == "scheme_b.pdf"

    doc_id = next(d["document_id"] for d in docs["documents"] if d["filename"] == "scheme_a.pdf")
    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    chat = client.post("/chat", json={"question": "eligibility small marginal farmers Aadhaar"}).json()
    assert all(s["document"] != "scheme_a.pdf" for s in chat["sources"])

    rebuilt = client.post("/index/rebuild").json()
    assert rebuilt["mode"] == "rebuild" and rebuilt["processed_pdfs"] == 1


def test_upload_rejects_non_pdf(client):
    r = _upload(client, "notes.txt", b"plain text").json()
    assert r["saved"] == 0 and r["results"][0]["status"] == "Rejected"


def test_chat_validation(client):
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422
    assert client.delete("/documents/..%2F..%2Fetc").status_code in (400, 404)


def test_followup_uses_condensed_question(settings, embeddings, pdf_a, pdf_b):
    llm = FakeListChatModel(responses=["drip irrigation subsidy small farmers", "Answer about drip [scheme_b.pdf, p. 1]"])
    svc = build_services(settings, faiss=FaissService(settings, embeddings=embeddings), llm=llm)
    with TestClient(create_app(svc)) as c:
        _upload(c, "scheme_a.pdf", pdf_a)
        _upload(c, "scheme_b.pdf", pdf_b)
        c.post("/ingest")
        r = c.post("/chat", json={
            "question": "How much is it?",
            "history": [{"role": "user", "content": "Tell me about micro irrigation"},
                        {"role": "assistant", "content": "Micro irrigation scheme gives drip subsidy."}],
        }).json()
    assert r["standalone_question"] == "drip irrigation subsidy small farmers"
    assert r["sources"][0]["document"] == "scheme_b.pdf"


def test_llm_failure_returns_502(settings, embeddings, pdf_a):
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise ConnectionError("network down")

    svc = build_services(settings, faiss=FaissService(settings, embeddings=embeddings), llm=Boom(responses=["x"]))
    with TestClient(create_app(svc)) as c:
        _upload(c, "scheme_a.pdf", pdf_a)
        c.post("/ingest")
        r = c.post("/chat", json={"question": "eligibility"})
    assert r.status_code == 502


def test_build_sources_dedupes_and_history_is_trimmed(client, pdf_a):
    _upload(client, "scheme_a.pdf", pdf_a)
    client.post("/ingest")
    chunks = retrieve(client.svc.faiss, "farmers", k=5)
    sources = build_sources(chunks + chunks)
    assert len(sources) == len({(s["document"], s["page"]) for s in sources})

    hist = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    assert format_history(hist, max_turns=2).count("User:") == 4


def test_prompt_enforces_grounding():
    p = SYSTEM_PROMPT.lower()
    assert "only from the context" in p
    assert "do not contain information" in p
    assert "never invent" in p
    assert "reporting" in p and "older" in p
