"""Pydantic request / response models (drive validation and Swagger docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    index_available: bool
    vector_count: int
    llm_configured: bool
    config_problems: list[str]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    checksum: str
    size_bytes: int
    status: str
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None
    uploaded_at: str
    indexed_at: str | None = None


class DocumentListResponse(BaseModel):
    count: int
    documents: list[DocumentInfo]


class UploadResult(BaseModel):
    filename: str
    status: str = Field(description="Uploaded | Duplicate | Rejected")
    detail: str
    document_id: str | None = None


class UploadResponse(BaseModel):
    saved: int
    results: list[UploadResult]


class DeleteResponse(BaseModel):
    document_id: str
    filename: str
    chunks_removed: int
    message: str


class DocumentIngestResult(BaseModel):
    document_id: str
    filename: str
    status: str
    pages: int | None = None
    chunks: int = 0
    empty_pages: int | None = None
    error: str | None = None


class IngestionSummary(BaseModel):
    mode: str
    status: str
    message: str
    uploaded_pdfs: int
    processed_pdfs: int
    pages_processed: int
    chunks_created: int
    vectors_indexed: int
    total_vectors_in_index: int
    index_location: str
    documents: list[DocumentIngestResult]


class IndexStatus(BaseModel):
    available: bool
    vector_count: int
    indexed_documents: int
    embedding_model: str
    index_embedding_model: str | None = None
    dimension: int | None = None
    index_path: str
    updated_at: str | None = None
    load_error: str | None = None
    last_indexing: dict


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000,
                          examples=["What are the eligibility criteria for this agriculture scheme?"])
    history: list[ChatTurn] = Field(default_factory=list, max_length=50,
                                    description="Recent chat turns (only the last few are used).")
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


class Source(BaseModel):
    document: str
    page: int
    relevant_content: str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    standalone_question: str | None = None
