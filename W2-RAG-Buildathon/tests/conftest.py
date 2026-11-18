"""Shared fixtures: isolated settings, fake embeddings / LLM, and a tiny PDF generator."""

from __future__ import annotations

import hashlib
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import build_settings  # noqa: E402


# ------------------------------------------------------------------ PDF generator
def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF where each string is one page ('' = blank page)."""
    objects: list[bytes] = []
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        lines = [ln for ln in text.split("\n")] if text else []
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
        for ln in lines:
            ops.append(f"({_escape(ln)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1") if lines else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# ------------------------------------------------------------------ fake models
_STOPWORDS = {"a", "an", "the", "is", "are", "for", "of", "and", "to", "in", "on", "what", "how", "this",
              "under", "with", "by", "it", "be", "can", "i", "do", "does", "there"}


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: shared words => higher cosine similarity."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in _STOPWORDS:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture()
def settings(tmp_path: Path):
    base = build_settings()
    return replace(
        base,
        watsonx_token="test-token",
        embedding_model="test/hash-embeddings",
        documents_dir=tmp_path / "documents",
        vectorstore_dir=tmp_path / "vectorstore",
        faiss_index_dir=tmp_path / "vectorstore" / "faiss_index",
        registry_path=tmp_path / "vectorstore" / "document_registry.json",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
    )


@pytest.fixture()
def embeddings() -> HashEmbeddings:
    return HashEmbeddings()


@pytest.fixture()
def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["Grounded answer citing [scheme_a.pdf, p. 2]."] * 50)


SCHEME_A = [
    "Kalaignar Agriculture Development Scheme 2023-24\nObjective: improve farm productivity in villages.",
    "Eligibility: small and marginal farmers owning less than 2 hectares of land are eligible.\n"
    "Required documents: Aadhaar card, land patta, bank passbook.",
    "",  # blank page -> must be skipped
]
SCHEME_B = [
    "Micro Irrigation Scheme\nSubsidy: 100 percent subsidy for drip irrigation for small farmers.",
    "How to apply: register online on the TNHORTNET portal and submit the application.",
]


@pytest.fixture()
def pdf_a() -> bytes:
    return make_pdf(SCHEME_A)


@pytest.fixture()
def pdf_b() -> bytes:
    return make_pdf(SCHEME_B)
