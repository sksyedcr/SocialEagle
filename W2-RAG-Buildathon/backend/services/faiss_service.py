"""Persistent FAISS vector store management.

Layout on disk (all application-generated)::

    vectorstore/faiss_index/
        index.faiss          # raw FAISS vectors
        index.pkl            # LangChain docstore + id mapping (pickle!)
        index_manifest.json  # embedding model, dimension, chunk ids per document

Security note on ``allow_dangerous_deserialization``
----------------------------------------------------
LangChain stores the FAISS docstore with Python ``pickle``. Unpickling can run
arbitrary code, so loading a malicious ``index.pkl`` is equivalent to running
an attacker's program. LangChain therefore refuses to load it unless
``allow_dangerous_deserialization=True`` is passed. We pass that flag ONLY for
the index directory this application writes itself (``settings.faiss_index_dir``),
we never accept index files over the API, and we verify the companion manifest
was written by this app before loading. Do not copy index files from untrusted
sources into ``vectorstore/``.
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import Settings
from utils.file_utils import atomic_write_json, read_json
from utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_NAME = "index_manifest.json"
APP_SIGNATURE = "tn-agri-schemes-rag"
MANIFEST_VERSION = 1


class IndexIncompatibleError(RuntimeError):
    """The persisted index was built with a different embedding configuration."""


def build_embeddings(model_name: str) -> Embeddings:
    """Create the HuggingFace sentence-transformer embedding model.

    Vectors are L2-normalised so inner-product search == cosine similarity.
    The same instance is used for indexing and querying.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model %s ...", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FaissService:
    def __init__(self, settings: Settings, embeddings: Embeddings | None = None) -> None:
        self.settings = settings
        self.index_dir: Path = settings.faiss_index_dir
        self._embeddings = embeddings
        self._store: FAISS | None = None
        self._manifest: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.load_error: str | None = None
        self.last_status: dict[str, Any] = {"status": "never_run", "at": None, "detail": None}

    # ------------------------------------------------------------------ embeddings
    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = build_embeddings(self.settings.embedding_model)
        return self._embeddings

    def _dimension(self) -> int:
        return len(self.embeddings.embed_query("dimension probe"))

    # ------------------------------------------------------------------ load / save
    @property
    def manifest_path(self) -> Path:
        return self.index_dir / MANIFEST_NAME

    def index_exists_on_disk(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def load(self) -> bool:
        """Load the persisted index if present and compatible. Returns True if loaded."""
        with self._lock:
            self.load_error = None
            if not self.index_exists_on_disk():
                logger.info("No persisted FAISS index at %s (will be created on first ingestion)", self.index_dir)
                self._store, self._manifest = None, {}
                return False
            manifest = read_json(self.manifest_path, default=None)
            try:
                self._check_manifest(manifest)
            except IndexIncompatibleError as exc:
                self.load_error = str(exc)
                logger.error("Refusing to load FAISS index: %s", exc)
                self._store, self._manifest = None, {}
                return False

            # Safe: this path is the app-owned index directory and the manifest
            # proves it was produced by this app (see module docstring).
            self._store = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            self._manifest = manifest
            logger.info(
                "Loaded FAISS index from disk: %d vectors, %d documents (no re-embedding needed)",
                self._store.index.ntotal,
                len(manifest.get("documents", {})),
            )
            return True

    def _check_manifest(self, manifest: dict[str, Any] | None) -> None:
        if not manifest or manifest.get("app") != APP_SIGNATURE:
            raise IndexIncompatibleError(
                "Index manifest missing or not created by this application. Use 'Rebuild Index'."
            )
        if manifest.get("embedding_model") != self.settings.embedding_model:
            raise IndexIncompatibleError(
                f"Index was built with embedding model '{manifest.get('embedding_model')}' but "
                f"EMBEDDING_MODEL is '{self.settings.embedding_model}'. Use 'Rebuild Index'."
            )

    def _new_manifest(self, dimension: int) -> dict[str, Any]:
        return {
            "app": APP_SIGNATURE,
            "version": MANIFEST_VERSION,
            "embedding_model": self.settings.embedding_model,
            "dimension": dimension,
            "distance": "cosine (normalized inner product)",
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "created_at": _now(),
            "updated_at": _now(),
            "documents": {},  # document_id -> {filename, checksum, chunk_ids}
        }

    def _persist(self, store: FAISS | None, manifest: dict[str, Any]) -> None:
        """Atomically replace the on-disk index with ``store``.

        The new index is written to a sibling temp directory first, then swapped
        in, so a crash mid-write never corrupts the existing index.
        """
        parent = self.index_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = parent / f"{self.index_dir.name}.tmp"
        bak_dir = parent / f"{self.index_dir.name}.bak"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(bak_dir, ignore_errors=True)

        if store is None or store.index.ntotal == 0:
            # Nothing left to search: remove the index entirely.
            if self.index_dir.exists():
                self.index_dir.rename(bak_dir)
                shutil.rmtree(bak_dir, ignore_errors=True)
            return

        manifest["updated_at"] = _now()
        manifest["vector_count"] = int(store.index.ntotal)
        store.save_local(str(tmp_dir))
        atomic_write_json(tmp_dir / MANIFEST_NAME, manifest)
        if self.index_dir.exists():
            self.index_dir.rename(bak_dir)
        tmp_dir.rename(self.index_dir)
        shutil.rmtree(bak_dir, ignore_errors=True)
        logger.info("Persisted FAISS index (%d vectors) to %s", store.index.ntotal, self.index_dir)

    # ------------------------------------------------------------------ mutations
    def indexed_document_ids(self) -> set[str]:
        return set(self._manifest.get("documents", {}))

    def add_documents(self, doc_id: str, filename: str, checksum: str, chunks: list[Document], ids: list[str]) -> int:
        """Incrementally add one document's chunks and persist. Idempotent per document."""
        if not chunks:
            return 0
        with self._lock:
            # Re-ingesting the same document replaces its old chunks (no duplicates).
            if doc_id in self.indexed_document_ids():
                self._delete_ids(self._manifest["documents"][doc_id]["chunk_ids"])
            if self._store is None:
                self._manifest = self._new_manifest(self._dimension())
                self._store = FAISS.from_documents(
                    chunks, self.embeddings, ids=ids, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                )
            else:
                self._store.add_documents(chunks, ids=ids)
            self._manifest["documents"][doc_id] = {"filename": filename, "checksum": checksum, "chunk_ids": ids}
            self._persist(self._store, self._manifest)
            return len(ids)

    def _delete_ids(self, ids: list[str]) -> None:
        if self._store is None or not ids:
            return
        existing = set(self._store.index_to_docstore_id.values())
        to_delete = [i for i in ids if i in existing]
        if to_delete:
            self._store.delete(ids=to_delete)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to ``doc_id`` and persist. Returns removed count."""
        with self._lock:
            entry = self._manifest.get("documents", {}).get(doc_id)
            if entry is None or self._store is None:
                return 0
            ids = entry["chunk_ids"]
            self._delete_ids(ids)
            del self._manifest["documents"][doc_id]
            if self._store.index.ntotal == 0:
                self._store = None
            self._persist(self._store, self._manifest)
            logger.info("Removed %d chunks of %s from FAISS", len(ids), doc_id)
            return len(ids)

    def rebuild(self, documents: list[tuple[str, str, str, list[Document], list[str]]]) -> int:
        """Replace the whole index with ``documents`` = [(doc_id, filename, checksum, chunks, ids)].

        Built fully in memory first; the old index is only replaced once the new
        one has been written successfully.
        """
        with self._lock:
            manifest = self._new_manifest(self._dimension())
            all_chunks: list[Document] = []
            all_ids: list[str] = []
            for doc_id, filename, checksum, chunks, ids in documents:
                if not chunks:
                    continue
                all_chunks.extend(chunks)
                all_ids.extend(ids)
                manifest["documents"][doc_id] = {"filename": filename, "checksum": checksum, "chunk_ids": ids}
            store = (
                FAISS.from_documents(
                    all_chunks, self.embeddings, ids=all_ids, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                )
                if all_chunks
                else None
            )
            self._persist(store, manifest)
            self._store, self._manifest = store, (manifest if store else {})
            self.load_error = None
            return len(all_ids)

    def record_status(self, status: str, detail: Any = None) -> None:
        self.last_status = {"status": status, "at": _now(), "detail": detail}

    # ------------------------------------------------------------------ search / status
    @property
    def is_ready(self) -> bool:
        return self._store is not None and self._store.index.ntotal > 0

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        with self._lock:
            if not self.is_ready:
                return []
            return self._store.similarity_search_with_score(query, k=k)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.is_ready,
                "vector_count": int(self._store.index.ntotal) if self._store else 0,
                "indexed_documents": len(self._manifest.get("documents", {})),
                "embedding_model": self.settings.embedding_model,
                "index_embedding_model": self._manifest.get("embedding_model"),
                "dimension": self._manifest.get("dimension"),
                "index_path": str(self.index_dir),
                "updated_at": self._manifest.get("updated_at"),
                "load_error": self.load_error,
                "last_indexing": self.last_status,
            }
