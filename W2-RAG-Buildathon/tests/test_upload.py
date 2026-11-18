"""Stage 3: upload validation, physical storage, duplicate handling, filename safety."""

from __future__ import annotations

from backend.services.document_service import DocumentService
from tests.conftest import make_pdf
from utils.file_utils import sanitize_filename


def test_save_upload_writes_to_documents_dir(settings, pdf_a):
    svc = DocumentService(settings)
    res = svc.save_upload("scheme_a.pdf", pdf_a)
    assert res["status"] == "Uploaded"
    saved = settings.documents_dir / "scheme_a.pdf"
    assert saved.exists() and saved.read_bytes() == pdf_a
    assert svc.list_documents()[0]["status"] == "Uploaded"


def test_duplicate_content_rejected(settings, pdf_a):
    svc = DocumentService(settings)
    svc.save_upload("scheme_a.pdf", pdf_a)
    res = svc.save_upload("copy_of_a.pdf", pdf_a)
    assert res["status"] == "Duplicate"
    assert len(svc.list_documents()) == 1


def test_same_name_different_content_gets_suffix(settings, pdf_a, pdf_b):
    svc = DocumentService(settings)
    svc.save_upload("scheme.pdf", pdf_a)
    res = svc.save_upload("scheme.pdf", pdf_b)
    assert res["status"] == "Uploaded"
    assert res["filename"] == "scheme_1.pdf"


def test_invalid_files_rejected(settings):
    svc = DocumentService(settings)
    assert svc.save_upload("notes.txt", b"hello")["status"] == "Rejected"
    assert svc.save_upload("empty.pdf", b"")["status"] == "Rejected"
    assert svc.save_upload("fake.pdf", b"PK\x03\x04 zip data")["status"] == "Rejected"
    assert list(settings.documents_dir.glob("*")) == []


def test_path_traversal_sanitized():
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("..\\..\\Windows\\evil.PDF") == "evil.pdf"
    assert sanitize_filename("Scheme <2024>?.pdf") == "Scheme _2024_.pdf"


def test_registry_persists_and_discovers_manual_files(settings, pdf_a, pdf_b):
    DocumentService(settings).save_upload("scheme_a.pdf", pdf_a)
    (settings.documents_dir / "dropped_in.pdf").write_bytes(pdf_b)
    names = {d["filename"] for d in DocumentService(settings).list_documents()}
    assert names == {"scheme_a.pdf", "dropped_in.pdf"}


def test_registry_drops_files_deleted_from_disk(settings, pdf_a):
    DocumentService(settings).save_upload("scheme_a.pdf", pdf_a)
    (settings.documents_dir / "scheme_a.pdf").unlink()
    assert DocumentService(settings).list_documents() == []


def test_upload_size_limit(settings):
    from dataclasses import replace
    svc = DocumentService(replace(settings, max_upload_mb=0))
    assert svc.save_upload("a.pdf", make_pdf(["hi there"]))["status"] == "Rejected"
