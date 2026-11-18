"""Streamlit frontend: Tamil Nadu Agriculture Schemes Assistant.

Left panel  - PDF upload & document / index management
Right panel - Chat interface

Run with:  streamlit run app.py   (backend must be running: uvicorn backend.main:app --reload)
Theme / spacing live in .streamlit/config.toml (no custom CSS).
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from utils.api_client import BackendClient, BackendError

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))
PANEL_HEIGHT = 640  # both columns share this height so they line up

st.set_page_config(
    page_title="TN Agriculture Schemes Assistant",
    page_icon=":material/agriculture:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# question text -> icon shown on the pill. Options are the plain questions, so the selected value
# is exactly what gets asked; icons are display-only via format_func.
SUGGESTED_QUESTIONS = {
    "What agriculture schemes are available?": ":material/list:",
    "What are the eligibility criteria?": ":material/how_to_reg:",
    "What subsidies are available for farmers?": ":material/payments:",
    "How can I apply for a particular scheme?": ":material/edit_document:",
    "What documents are required?": ":material/folder_shared:",
    "Compare two agriculture schemes.": ":material/compare_arrows:",
}

STATUS_BADGE = {
    "Indexed": ("green", ":material/check_circle:"),
    "Uploaded": ("orange", ":material/schedule:"),
    "Failed": ("red", ":material/error:"),
}


# ---------------------------------------------------------------- state
@st.cache_resource
def get_client() -> BackendClient:
    return BackendClient(BACKEND_URL)


client = get_client()
ss = st.session_state
ss.setdefault("messages", [])        # [{role, content, sources?}]
ss.setdefault("uploader_key", 0)      # bump to reset the file_uploader after upload
ss.setdefault("flash", None)          # (kind, message) shown once after a rerun
ss.setdefault("last_summary", None)


def flash(kind: str, message: str) -> None:
    ss.flash = (kind, message)


def safe_call(fn, *args, **kwargs) -> Any | None:
    try:
        return fn(*args, **kwargs)
    except BackendError as exc:
        st.toast(str(exc), icon=":material/cloud_off:")
        return None


# ---------------------------------------------------------------- header
with st.container(horizontal=True, vertical_alignment="center"):
    st.title(":material/agriculture: Tamil Nadu Agriculture Schemes Assistant", anchor=False)
st.caption("Upload Government scheme PDFs and ask grounded questions. "
           "Every answer cites the source PDF and page.")

backend_error: str | None = None
health = None
try:
    health = client.health()
except BackendError as exc:
    backend_error = str(exc)

if backend_error:
    st.warning(backend_error + " If you just started or edited the backend, it may still be loading the "
               "embedding model; wait a few seconds and press **Retry**.",
               title="Backend not ready", icon=":material/hourglass_top:")
    st.button("Retry", icon=":material/refresh:", type="primary")
elif not health["llm_configured"]:
    st.warning(" ".join(health["config_problems"]), title="LLM not configured", icon=":material/key_off:")

if ss.flash:
    kind, msg = ss.flash
    getattr(st, kind)(msg)
    ss.flash = None

backend_up = health is not None
docs_resp = safe_call(client.list_documents) if backend_up else None
docs: list[dict[str, Any]] = docs_resp["documents"] if docs_resp else []
index = safe_call(client.index_status) if backend_up else None

left, right = st.columns([0.36, 0.64], gap="medium")


# ================================================================ LEFT PANEL
def render_summary(summary: dict[str, Any]) -> None:
    with st.expander(f"Last indexing summary: {summary['status']}", icon=":material/summarize:"):
        st.caption(summary["message"])
        with st.grid(2):
            st.metric("PDFs processed", f"{summary['processed_pdfs']}/{summary['uploaded_pdfs']}", border=True)
            st.metric("Pages", summary["pages_processed"], border=True)
            st.metric("Chunks created", summary["chunks_created"], border=True)
            st.metric("Vectors indexed", summary["vectors_indexed"], border=True)
        st.caption(f"Total vectors in index: **{summary['total_vectors_in_index']}**  \n"
                   f"Index location: `{summary['index_location']}`")
        for d in summary["documents"]:
            if d["status"] == "Failed":
                st.error(f"{d['filename']}: {d['error']}", icon=":material/error:")


with left:
    with st.container(height=PANEL_HEIGHT, border=True):
        # -------- Upload
        st.subheader(":material/upload_file: Upload PDFs", anchor=False)
        uploaded = st.file_uploader(
            "Agriculture scheme PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{ss.uploader_key}",
            help="Files are saved to the project's documents/ folder before indexing.",
            label_visibility="collapsed",
            disabled=not backend_up,
        )
        if st.button("Upload & process documents", type="primary", icon=":material/rocket_launch:",
                     width="stretch", disabled=not backend_up):
            if uploaded:
                with st.spinner(f"Uploading {len(uploaded)} file(s)..."):
                    up = safe_call(client.upload, [(f.name, f.getvalue()) for f in uploaded])
                if up:
                    for r in up["results"]:
                        if r["status"] == "Rejected":
                            st.toast(f"{r['filename']}: {r['detail']}", icon=":material/warning:")
                        elif r["status"] == "Duplicate":
                            st.toast(r["detail"], icon=":material/info:")
            with st.spinner("Extracting text, embedding and indexing..."):
                summary = safe_call(client.ingest)
            if summary:
                ss.last_summary = summary
                ss.uploader_key += 1
                kind = {"success": "success", "up_to_date": "info", "partial": "warning"}.get(
                    summary["status"], "error")
                flash(kind, summary["message"])
                st.rerun()

        st.divider()

        # -------- Documents
        st.subheader(f":material/folder_open: Documents ({len(docs) if backend_up else '-'})", anchor=False)
        if not backend_up:
            st.caption("Waiting for the backend...")
        elif not docs:
            st.caption("No documents uploaded yet.")
        for d in docs:
            color, icon = STATUS_BADGE.get(d["status"], ("gray", ":material/help:"))
            with st.container(border=True):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f":material/picture_as_pdf: **{d['filename']}**")
                    with st.popover("", icon=":material/delete:", help="Delete document", type="tertiary"):
                        st.write(f"Delete **{d['filename']}** and remove it from the index?")
                        if st.button("Delete", key=f"del_{d['document_id']}", type="primary",
                                     icon=":material/delete:"):
                            with st.spinner("Deleting and updating index..."):
                                res = safe_call(client.delete, d["document_id"])
                            if res:
                                flash("success", res["message"])
                                st.rerun()
                meta = [f":{color}-badge[{icon} {d['status']}]"]
                if d.get("pages"):
                    meta.append(f"{d['pages']} pages")
                if d.get("chunks"):
                    meta.append(f"{d['chunks']} chunks")
                st.caption(" · ".join(meta))
                if d["status"] == "Failed" and d.get("error"):
                    st.caption(f":red[{d['error']}]")

        st.divider()

        # -------- FAISS index
        st.subheader(":material/database: FAISS index", anchor=False)
        if index:
            with st.grid(2):
                st.metric("Status", "Ready" if index["available"] else "Empty", border=True)
                st.metric("Vectors", index["vector_count"], border=True)
                st.metric("Indexed docs", index["indexed_documents"], border=True)
                st.metric("Total docs", len(docs), border=True)
            st.caption(f"Embedding model: `{index['embedding_model']}`"
                       + (f"  \nLast updated: {index['updated_at']}" if index.get("updated_at") else ""))
            if index.get("load_error"):
                st.error(index["load_error"], icon=":material/warning:")
        if st.button("Rebuild FAISS index", icon=":material/autorenew:", width="stretch",
                     disabled=not (backend_up and docs)):
            with st.spinner("Rebuilding the full index from documents/ ..."):
                summary = safe_call(client.rebuild)
            if summary:
                ss.last_summary = summary
                flash("success" if summary["status"] == "success" else "warning", summary["message"])
                st.rerun()

        if ss.last_summary:
            render_summary(ss.last_summary)


# ================================================================ RIGHT PANEL
def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/menu_book:"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**{i}. {s['document']}** · page {s['page']}")
            st.caption(s["relevant_content"])


with right:
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(":material/forum: Chat", anchor=False)
            st.space("stretch")
            if st.button("Clear chat", icon=":material/delete_sweep:", disabled=not ss.messages):
                ss.messages = []
                st.rerun()

        # Fixed height (minus header + input) so the chat card matches the left panel.
        chat_box = st.container(height=PANEL_HEIGHT - 150, border=False)
        prompt = st.chat_input("Ask about schemes, eligibility, subsidies, how to apply...",
                               disabled=not backend_up)

        with chat_box:
            if not ss.messages:
                with st.chat_message("assistant", avatar=":material/agriculture:"):
                    st.markdown("**Vanakkam!** I answer questions about Tamil Nadu agriculture schemes using "
                                "**only the PDFs you upload**, with page citations. Try one of these:")
                    suggestions = st.empty()
                    picked = suggestions.pills("Suggested questions", list(SUGGESTED_QUESTIONS),
                                               format_func=lambda q: f"{SUGGESTED_QUESTIONS[q]} {q}",
                                               label_visibility="collapsed", disabled=not backend_up)
                    if picked and not prompt:
                        prompt = picked
                    if prompt:
                        suggestions.empty()
            for m in ss.messages:
                with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user"
                                     else ":material/agriculture:"):
                    st.markdown(m["content"])
                    render_sources(m.get("sources", []))

            if prompt:
                # Only the last few turns (text only, no sources) go to the backend.
                history = [{"role": m["role"], "content": m["content"]} for m in ss.messages][-HISTORY_TURNS * 2:]
                ss.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar=":material/agriculture:"):
                    with st.spinner("Searching documents and generating answer..."):
                        try:
                            resp = client.chat(prompt, history)
                            answer, sources = resp["answer"], resp["sources"]
                        except BackendError as exc:
                            answer, sources = f":red[**Error:** {exc}]", []
                    st.markdown(answer)
                    render_sources(sources)
                ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
                st.rerun()
