"""Fetch / retrieval pipeline: query -> FAISS top-k -> formatted context + prompts.

Kept independent from the LLM so retrieval and citations can be tested
without network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are the "Tamil Nadu Agriculture Schemes Assistant". You help farmers, \
officials and citizens understand Tamil Nadu Government agriculture schemes using ONLY the \
document excerpts provided in the CONTEXT.

Strict rules:
1. Answer ONLY from the CONTEXT. Never use outside knowledge and never invent scheme names, \
amounts, percentages, dates, eligibility rules, deadlines, offices or URLs.
2. If the CONTEXT does not contain the answer, say clearly: "The uploaded documents do not \
contain information about this." You may then mention what related information IS available.
3. Cite every factual statement with its source in the form [filename, p. N]. Use the exact \
filename and page shown in the CONTEXT headers.
4. Keep information from different documents and different reporting / financial years distinct. \
Say which document and year each detail comes from. Do not merge figures across documents.
5. Do not present older scheme details as current. If a document refers to a specific year \
(e.g. 2022-23), state that year, and note that details may have changed if no newer source is given.
6. When comparing schemes, use a short table or side-by-side bullets, citing each cell or bullet.
7. Be concise and well structured: use headings or bullet points for objectives, eligibility, \
benefits/subsidies, application procedure and required documents where relevant.
8. Answer in the language of the question when possible."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "RECENT CONVERSATION (for resolving references only, not a source of facts):\n"
            "{history}\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "Answer using only the CONTEXT and cite sources as [filename, p. N].",
        ),
    ]
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message into a single standalone search query about Tamil Nadu "
            "agriculture schemes, resolving pronouns and references (e.g. 'it', 'that scheme', 'the second "
            "one') using the conversation. If it is already standalone, return it unchanged. Output ONLY "
            "the rewritten query, nothing else.",
        ),
        ("human", "Conversation:\n{history}\n\nLatest message: {question}\n\nStandalone query:"),
    ]
)

NO_CONTEXT_ANSWER = (
    "I could not find relevant information in the uploaded documents to answer this question. "
    "Please upload and process the relevant scheme PDFs, or rephrase your question."
)


class SearchBackend(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Document, float]]: ...


@dataclass
class RetrievedChunk:
    document: str
    page: int
    document_id: str
    content: str
    score: float


def retrieve(backend: SearchBackend, query: str, k: int, min_score: float = 0.0) -> list[RetrievedChunk]:
    """Embed ``query`` (inside the backend, same model as indexing) and return top-k chunks."""
    results = backend.search(query, k=k)
    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        if score < min_score:
            continue
        chunks.append(
            RetrievedChunk(
                document=str(doc.metadata.get("source", "unknown")),
                page=int(doc.metadata.get("page", 0)),
                document_id=str(doc.metadata.get("document_id", "")),
                content=doc.page_content,
                score=float(score),
            )
        )
    return chunks


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [
        f"[Excerpt {i}] Source: {c.document}, page {c.page}\n{c.content}" for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def format_history(history: list[dict[str, str]], max_turns: int, max_chars: int = 600) -> str:
    """Compact the last ``max_turns`` Q/A pairs. Long answers are truncated to save tokens."""
    if not history or max_turns <= 0:
        return "(none)"
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        text = (msg.get("content") or "").strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_sources(chunks: list[RetrievedChunk], max_chars: int = 700) -> list[dict[str, object]]:
    """Unique (document, page) citations in retrieval order."""
    seen: set[tuple[str, int]] = set()
    sources: list[dict[str, object]] = []
    for c in chunks:
        key = (c.document, c.page)
        if key in seen:
            continue
        seen.add(key)
        content = c.content if len(c.content) <= max_chars else c.content[:max_chars].rstrip() + " ..."
        sources.append(
            {"document": c.document, "page": c.page, "relevant_content": content, "score": round(c.score, 4)}
        )
    return sources
