"""Safe file handling helpers: filename sanitisation, checksums, PDF validation, atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

PDF_MAGIC = b"%PDF-"
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")


def sanitize_filename(name: str) -> str:
    """Strip any path components and unsafe characters while keeping the name readable."""
    name = unicodedata.normalize("NFKC", name or "")
    name = name.replace("\\", "/").split("/")[-1]  # drop directories / traversal
    name = _SAFE_CHARS.sub("_", name).strip(" .")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, "pdf"
    stem = stem[:150] or "document"
    return f"{stem}.{ext.lower()}"


def unique_path(directory: Path, filename: str) -> Path:
    """Return ``directory/filename``, appending ``_1``, ``_2``... if it already exists."""
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf_bytes(filename: str, data: bytes, max_bytes: int) -> str | None:
    """Return an error message if the payload is not an acceptable PDF, else None."""
    if not filename.lower().endswith(".pdf"):
        return "Only .pdf files are accepted."
    if not data:
        return "File is empty."
    if len(data) > max_bytes:
        return f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit."
    # The PDF header must appear within the first 1 KB per the PDF spec.
    if PDF_MAGIC not in data[:1024]:
        return "File content is not a valid PDF (missing %PDF header)."
    return None


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + rename so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
