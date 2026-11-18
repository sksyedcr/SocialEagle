"""Upload / ingestion pipeline: PDF -> pages -> chunks (with metadata).

Pure functions with no FAISS or FastAPI dependencies, so they are easy to test
and reuse. Embedding + indexing is done by ``FaissService``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.logger import get_logger

logger = get_logger(__name__)

_WS = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class PDFLoadError(RuntimeError):
    """The PDF could not be parsed or contains no extractable text."""


@dataclass
class LoadedDocument:
    document_id: str
    filename: str
    total_pages: int
    pages: list[Document]  # only pages with extractable text
    empty_pages: list[int] = field(default_factory=list)  # 1-based page numbers


@dataclass
class ChunkedDocument:
    document_id: str
    filename: str
    total_pages: int
    empty_pages: list[int]
    chunks: list[Document]
    chunk_ids: list[str]


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load_pdf(path: Path, document_id: str, filename: str | None = None) -> LoadedDocument:
    """Load a PDF page by page with PyPDFLoader, attaching citation metadata.

    ``page`` in the returned metadata is the 1-based physical page number
    (what a human sees in a PDF viewer). Empty pages (e.g. scanned images
    without OCR) are skipped and reported.
    """
    from langchain_community.document_loaders import PyPDFLoader

    filename = filename or path.name
    try:
        raw_pages = PyPDFLoader(str(path)).load()
    except Exception as exc:  # pypdf raises a variety of error types for malformed files
        raise PDFLoadError(f"Could not read PDF '{filename}': {exc}") from exc

    pages: list[Document] = []
    empty: list[int] = []
    for idx, page in enumerate(raw_pages):
        page_no = int(page.metadata.get("page", idx)) + 1
        text = _clean(page.page_content or "")
        if len(text) < 5:
            empty.append(page_no)
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": page_no,
                    "document_id": document_id,
                    "total_pages": len(raw_pages),
                },
            )
        )

    if not raw_pages:
        raise PDFLoadError(f"'{filename}' has no pages.")
    if not pages:
        raise PDFLoadError(
            f"'{filename}' has {len(raw_pages)} page(s) but no extractable text "
            "(it may be a scanned image PDF that needs OCR)."
        )
    if empty:
        logger.info("%s: skipped %d empty page(s): %s", filename, len(empty), empty[:20])
    return LoadedDocument(document_id, filename, len(raw_pages), pages, empty)


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def split_document(loaded: LoadedDocument, chunk_size: int, chunk_overlap: int) -> ChunkedDocument:
    """Split page Documents into chunks. Each chunk keeps its page's metadata.

    Splitting is done per page so that every chunk maps to exactly one page
    number, which keeps citations precise.
    """
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(loaded.pages)
    ids: list[str] = []
    per_page_counter: dict[int, int] = {}
    for chunk in chunks:
        page = chunk.metadata["page"]
        n = per_page_counter.get(page, 0)
        per_page_counter[page] = n + 1
        chunk_id = f"{loaded.document_id}:p{page}:c{n}"
        chunk.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)
    validate_chunks(chunks)
    return ChunkedDocument(loaded.document_id, loaded.filename, loaded.total_pages, loaded.empty_pages, chunks, ids)


REQUIRED_METADATA = ("source", "page", "document_id", "chunk_id")


def validate_chunks(chunks: list[Document]) -> None:
    """Fail fast if any chunk is missing the metadata needed for citations."""
    for chunk in chunks:
        missing = [k for k in REQUIRED_METADATA if chunk.metadata.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Chunk missing metadata {missing}: {chunk.metadata}")
        if not isinstance(chunk.metadata["page"], int) or chunk.metadata["page"] < 1:
            raise ValueError(f"Invalid page number in chunk metadata: {chunk.metadata}")


def process_pdf(path: Path, document_id: str, filename: str, chunk_size: int, chunk_overlap: int) -> ChunkedDocument:
    """Convenience: load + split one PDF."""
    return split_document(load_pdf(path, document_id, filename), chunk_size, chunk_overlap)
