"""Thin HTTP client used by the Streamlit frontend to talk to the FastAPI backend."""

from __future__ import annotations

from typing import Any

import requests


class BackendError(RuntimeError):
    pass


class BackendClient:
    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, timeout: int | None = None, **kwargs: Any) -> Any:
        try:
            resp = requests.request(method, f"{self.base_url}{path}", timeout=timeout or self.timeout, **kwargs)
        except requests.ConnectionError as exc:
            raise BackendError(
                f"Cannot reach the backend at {self.base_url}. Start it with: uvicorn backend.main:app --reload"
            ) from exc
        except requests.Timeout as exc:
            raise BackendError("The backend took too long to respond. Please try again.") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            if isinstance(detail, list):  # pydantic validation errors
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise BackendError(f"{detail}")
        return resp.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=5)

    def list_documents(self) -> dict[str, Any]:
        return self._request("GET", "/documents")

    def upload(self, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        payload = [("files", (name, data, "application/pdf")) for name, data in files]
        return self._request("POST", "/documents/upload", files=payload, timeout=120)

    def delete(self, document_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/documents/{document_id}", timeout=300)

    def ingest(self) -> dict[str, Any]:
        return self._request("POST", "/ingest", timeout=1800)

    def rebuild(self) -> dict[str, Any]:
        return self._request("POST", "/index/rebuild", timeout=1800)

    def index_status(self) -> dict[str, Any]:
        return self._request("GET", "/index/status", timeout=10)

    def chat(self, question: str, history: list[dict[str, str]]) -> dict[str, Any]:
        return self._request("POST", "/chat", json={"question": question, "history": history}, timeout=300)
