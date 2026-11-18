"""Document storage + registry.

Uploaded PDFs are written physically to ``documents/``. A JSON registry
(``vectorstore/document_registry.json``) tracks each document's id, filename,
SHA-256 checksum, status and indexing statistics. The checksum is the
fingerprint used to prevent duplicate uploads / duplicate indexing.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from config.settings import Settings
from utils.file_utils import (
    atomic_write_json,
    is_within,
    read_json,
    sanitize_filename,
    sha256_bytes,
    sha256_file,
    unique_path,
    validate_pdf_bytes,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DocStatus(str, Enum):
    UPLOADED = "Uploaded"
    INDEXED = "Indexed"
    FAILED = "Failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentService:
    """Thread-safe registry of uploaded PDFs."""

    REGISTRY_VERSION = 1

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents_dir: Path = settings.documents_dir
        self.registry_path: Path = settings.registry_path
        self._lock = threading.RLock()
        settings.ensure_dirs()
        self._docs: dict[str, dict[str, Any]] = self._load()
        self.reconcile()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = read_json(self.registry_path, default={})
        except Exception:  # corrupt registry -> start fresh, a rebuild will repair the index
            logger.exception("Document registry unreadable; starting with an empty registry")
            return {}
        return dict(data.get("documents", {}))

    def _save(self) -> None:
        atomic_write_json(
            self.registry_path,
            {"version": self.REGISTRY_VERSION, "documents": self._docs},
        )

    def reconcile(self) -> None:
        """Sync the registry with what is physically on disk.

        * Registry entries whose file has vanished are dropped.
        * PDFs dropped manually into ``documents/`` are registered as Uploaded.
        """
        with self._lock:
            changed = False
            for doc_id, doc in list(self._docs.items()):
                if not (self.documents_dir / doc["filename"]).exists():
                    logger.warning("File for %s (%s) missing on disk; removing from registry", doc_id, doc["filename"])
                    del self._docs[doc_id]
                    changed = True
            known = {d["filename"] for d in self._docs.values()}
            for pdf in sorted(self.documents_dir.glob("*.pdf")):
                if pdf.name in known:
                    continue
                checksum = sha256_file(pdf)
                doc_id = checksum[:16]
                if doc_id in self._docs:  # same content under another name -> ignore duplicate
                    logger.warning("Skipping %s: duplicate content of %s", pdf.name, self._docs[doc_id]["filename"])
                    continue
                self._docs[doc_id] = self._new_entry(doc_id, pdf.name, checksum, pdf.stat().st_size)
                changed = True
                logger.info("Discovered PDF on disk: %s", pdf.name)
            if changed:
                self._save()

    @staticmethod
    def _new_entry(doc_id: str, filename: str, checksum: str, size: int) -> dict[str, Any]:
        return {
            "document_id": doc_id,
            "filename": filename,
            "checksum": checksum,
            "size_bytes": size,
            "status": DocStatus.UPLOADED.value,
            "pages": None,
            "chunks": None,
            "error": None,
            "uploaded_at": _now(),
            "indexed_at": None,
        }

    # ------------------------------------------------------------------ queries
    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((dict(d) for d in self._docs.values()), key=lambda d: d["uploaded_at"])

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            doc = self._docs.get(doc_id)
            return dict(doc) if doc else None

    def path_for(self, doc_id: str) -> Path:
        doc = self._docs[doc_id]
        path = (self.documents_dir / doc["filename"]).resolve()
        if not is_within(path, self.documents_dir):  # defence in depth
            raise ValueError("Resolved document path escapes the documents directory")
        return path

    def pending_ids(self) -> list[str]:
        """Documents not yet successfully indexed (Uploaded or previously Failed)."""
        with self._lock:
            return [i for i, d in self._docs.items() if d["status"] != DocStatus.INDEXED.value]

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._docs)

    # ------------------------------------------------------------------ mutations
    def save_upload(self, original_name: str, data: bytes) -> dict[str, Any]:
        """Validate and persist one uploaded file. Returns a per-file result dict."""
        safe_name = sanitize_filename(original_name)
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        error = validate_pdf_bytes(safe_name, data, max_bytes)
        if error:
            logger.warning("Rejected upload %r: %s", original_name, error)
            return {"filename": original_name, "status": "Rejected", "detail": error, "document_id": None}

        checksum = sha256_bytes(data)
        doc_id = checksum[:16]
        with self._lock:
            if doc_id in self._docs:
                existing = self._docs[doc_id]
                return {
                    "filename": existing["filename"],
                    "status": "Duplicate",
                    "detail": f"Identical content already uploaded as '{existing['filename']}'.",
                    "document_id": doc_id,
                }
            target = unique_path(self.documents_dir, safe_name)
            target.write_bytes(data)
            self._docs[doc_id] = self._new_entry(doc_id, target.name, checksum, len(data))
            self._save()
        logger.info("Saved upload %s -> %s (%d bytes)", original_name, target.name, len(data))
        detail = "Saved." if target.name == safe_name else f"Saved as '{target.name}' (name already in use)."
        return {"filename": target.name, "status": DocStatus.UPLOADED.value, "detail": detail, "document_id": doc_id}

    def mark_indexed(self, doc_id: str, pages: int, chunks: int) -> None:
        with self._lock:
            if doc_id in self._docs:
                self._docs[doc_id].update(
                    status=DocStatus.INDEXED.value, pages=pages, chunks=chunks, error=None, indexed_at=_now()
                )
                self._save()

    def mark_failed(self, doc_id: str, error: str, pages: int | None = None) -> None:
        with self._lock:
            if doc_id in self._docs:
                self._docs[doc_id].update(status=DocStatus.FAILED.value, error=error, pages=pages, chunks=0)
                self._save()

    def mark_all_uploaded(self) -> None:
        """Reset every document to Uploaded (used before a full rebuild)."""
        with self._lock:
            for doc in self._docs.values():
                doc.update(status=DocStatus.UPLOADED.value, chunks=None, error=None, indexed_at=None)
            self._save()

    def delete(self, doc_id: str) -> dict[str, Any] | None:
        """Remove the PDF from disk and the registry. Returns the removed entry."""
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                return None
            path = self.path_for(doc_id)
            path.unlink(missing_ok=True)
            del self._docs[doc_id]
            self._save()
        logger.info("Deleted document %s (%s)", doc_id, doc["filename"])
        return doc
