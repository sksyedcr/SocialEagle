"""Render app.py headlessly against a fake backend (0, 2 and 6 docs) to exercise every UI branch."""
import os
import pytest
import sys, warnings
warnings.simplefilter("ignore")
from streamlit.testing.v1 import AppTest
import utils.api_client as api

def make_docs(n):
    st = ["Indexed", "Uploaded", "Failed"]
    return [{"document_id": f"id{i:014d}", "filename": f"scheme_{i}.pdf", "checksum": "x", "size_bytes": 10,
             "status": st[i % 3], "pages": 4 if i % 3 == 0 else None, "chunks": 9 if i % 3 == 0 else None,
             "error": "no extractable text" if i % 3 == 2 else None, "uploaded_at": f"2026-11-18T00:00:0{i}",
             "indexed_at": None} for i in range(n)]

def patch(n):
    docs = make_docs(n)
    api.BackendClient.health = lambda self: {"status": "ok", "index_available": n > 0, "vector_count": 9 * n,
                                             "llm_configured": True, "config_problems": []}
    api.BackendClient.list_documents = lambda self: {"count": n, "documents": docs}
    api.BackendClient.index_status = lambda self: {"available": n > 0, "vector_count": 9 * n, "indexed_documents": n,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "index_embedding_model": None, "dimension": 384,
        "index_path": "vectorstore/faiss_index", "updated_at": "2026-11-18T00:00:00", "load_error": None,
        "last_indexing": {}}
    api.BackendClient.chat = lambda self, q, h: {"answer": f"Answer to {q} [scheme_0.pdf, p. 2]",
        "sources": [{"document": "scheme_0.pdf", "page": 2, "relevant_content": "Eligibility: small farmers", "score": 0.8}]}


APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.mark.parametrize("n", [0, 2, 6])
def test_ui_renders_without_exceptions(n):
    patch(n)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    if n == 6:
        at.chat_input[0].set_value("What subsidies are available?").run()
        assert not at.exception
        names = [m.name for m in at.chat_message]
        assert names.count("user") == 1 and names.count("assistant") == 1


def test_ui_backend_down_shows_not_ready():
    patch(0)

    def down(self):
        raise api.BackendError("The backend took too long to respond. Please try again.")

    api.BackendClient.health = down
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("too long" in w.value for w in at.warning)
    assert at.chat_input[0].disabled


def test_ui_suggestion_pill_asks_question():
    patch(2)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.pills[0].set_value("What subsidies are available for farmers?").run()
    assert not at.exception, [e.value for e in at.exception]
    names = [m.name for m in at.chat_message]
    assert names.count("user") == 1 and names.count("assistant") == 1
    assert at.session_state["messages"][0]["content"] == "What subsidies are available for farmers?"
