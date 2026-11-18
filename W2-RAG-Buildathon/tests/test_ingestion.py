"""Stage 4/5: PDF loading, page metadata, splitting, invalid PDF handling."""

from __future__ import annotations

import pytest

from pipelines.ingestion_pipeline import PDFLoadError, load_pdf, process_pdf, validate_chunks
from tests.conftest import make_pdf


def test_load_pdf_page_metadata(tmp_path, pdf_a):
    path = tmp_path / "scheme_a.pdf"
    path.write_bytes(pdf_a)
    loaded = load_pdf(path, "doc123", "scheme_a.pdf")
    assert loaded.total_pages == 3
    assert [p.metadata["page"] for p in loaded.pages] == [1, 2]  # 1-based, blank page 3 skipped
    assert loaded.empty_pages == [3]
    assert all(p.metadata["source"] == "scheme_a.pdf" for p in loaded.pages)
    assert all(p.metadata["document_id"] == "doc123" for p in loaded.pages)
    assert "Eligibility" in loaded.pages[1].page_content


def test_split_keeps_metadata_and_unique_ids(tmp_path):
    long_text = "\n".join(f"Line {i}: farmers receive subsidy support under this scheme." for i in range(60))
    path = tmp_path / "long.pdf"
    path.write_bytes(make_pdf([long_text, "Short second page about eligibility."]))
    chunked = process_pdf(path, "docX", "long.pdf", chunk_size=300, chunk_overlap=50)
    assert len(chunked.chunks) > 2
    assert len(set(chunked.chunk_ids)) == len(chunked.chunk_ids)
    assert all(len(c.page_content) <= 300 for c in chunked.chunks)
    assert {c.metadata["page"] for c in chunked.chunks} == {1, 2}
    validate_chunks(chunked.chunks)


def test_blank_pdf_raises(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf(["", ""]))
    with pytest.raises(PDFLoadError, match="no extractable text"):
        load_pdf(path, "d", "blank.pdf")


def test_corrupt_pdf_raises(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(PDFLoadError):
        load_pdf(path, "d", "bad.pdf")
