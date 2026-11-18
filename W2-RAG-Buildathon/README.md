# 🌾 Tamil Nadu Agriculture Schemes Assistant — Persistent FAISS RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about Tamil Nadu Government
agriculture schemes **using only the PDFs you upload**, with filename + page citations.

| Layer | Technology |
|---|---|
| Frontend | Streamlit (two-panel: documents left, chat right) |
| Backend | FastAPI (+ Swagger at `/docs`) |
| PDF processing | LangChain `PyPDFLoader` (pypdf) |
| Splitting | `RecursiveCharacterTextSplitter` (1000 / 150) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS (LangChain), persisted to local disk |
| LLM | IBM watsonx.ai MaaS **GPT-OSS-120B** via LangChain `ChatOpenAI` (OpenAI-compatible endpoint) |

---

## Architecture

```
Streamlit (app.py)
 ├─ Left : upload / list / delete PDFs, rebuild index, index status
 └─ Right: chat (st.chat_message / st.chat_input) + expandable sources
        │  HTTP (utils/api_client.py)
        ▼
FastAPI (backend/main.py)
 ├─ /documents/*  → DocumentService  → documents/*.pdf + vectorstore/document_registry.json
 ├─ /ingest, /index/*  → RAGService.ingest → pipelines/ingestion_pipeline.py
 │                          PyPDFLoader → clean pages → split → FaissService (embed + persist)
 └─ /chat → RAGService.answer → (condense follow-up) → pipelines/retrieval_pipeline.py
                                FAISS top-k → grounded prompt → watsonx.ai GPT-OSS-120B
```

```
W2-RAG-Buildathon/
├── app.py                       # Streamlit UI
├── backend/
│   ├── main.py                  # FastAPI app, startup validation, index load
│   ├── dependencies.py          # service container
│   ├── schemas.py               # Pydantic models
│   ├── routes/  chat.py  ingestion.py  documents.py
│   └── services/ rag_service.py  faiss_service.py  document_service.py
├── pipelines/ ingestion_pipeline.py  retrieval_pipeline.py
├── config/settings.py           # env-driven settings + validation
├── utils/ logger.py  file_utils.py  api_client.py
├── documents/                   # uploaded PDFs (git-ignored)
├── vectorstore/                 # faiss_index/ + document_registry.json (git-ignored)
├── tests/                       # offline pytest suite
├── .env.example  .gitignore  requirements.txt
```

---

## 1. Environment setup

Requires Python 3.10+ (tested on 3.14, Windows 11).

```bash
cd W2-RAG-Buildathon
python -m venv .venv
# Windows (Git Bash):  source .venv/Scripts/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate
pip install -r requirements.txt
```

> First install downloads PyTorch (~200 MB+). The embedding model (~90 MB) downloads on the
> first ingestion and is cached by HuggingFace afterwards.

## 2. watsonx.ai credentials

```bash
cp .env.example .env      # then edit .env
```

```ini
WATSONX_TOKEN=<your MaaS API token>
LLM_BASE_URL=https://maas.apps.wdc04-prod-ocp-01.cl.sec.ibm.com/models/gpt-oss-120b/v1
LLM_MODEL=gpt-oss-120b
```

* Legacy names `WATSONX_BASE_URL` / `WATSONX_MODEL_NAME` are also accepted.
* The token is read from the environment only — it is never logged, printed or returned by the API.
* On startup the backend validates the configuration; problems are logged and shown in the UI
  (`/health → config_problems`). There is **no silent fallback** to another LLM provider.
* `.env` is git-ignored. **Never commit it.** Verify with `git check-ignore -v .env`.

## 3. Start both applications (two terminals, from `W2-RAG-Buildathon/`)

```bash
# Terminal 1 – Backend (reload only watches code folders, not documents/ or vectorstore/)
uvicorn backend.main:app --reload --reload-dir backend --reload-dir pipelines --reload-dir config --reload-dir utils
#   → Swagger UI: http://127.0.0.1:8000/docs
#   Each (re)start loads the embedding model first; the UI shows "Backend not ready" until it's up.

# Terminal 2 – Frontend
streamlit run app.py
#   → http://localhost:8501
```

## 4. PDF upload workflow & initial ingestion

1. In the **left panel**, choose one or more PDFs.
2. Click **Upload & Process Documents**:
   * Each file is validated (`.pdf` extension, non-empty, `%PDF` header, size ≤ `MAX_UPLOAD_MB`).
   * It is **saved physically to `documents/`** (filename sanitised; name clashes get `_1`, `_2`…).
   * A SHA-256 checksum is recorded; identical content is reported as *Duplicate* and not re-indexed.
   * `POST /ingest` then loads each *pending* PDF page by page, skips empty pages, splits into
     chunks with `source`, `page` (1-based), `document_id`, `chunk_id` metadata, embeds them and
     adds them to FAISS, which is saved to disk.
3. The summary shows PDFs, pages, chunks, vectors indexed and the index location. Each document
   shows **Uploaded / Indexed / Failed** with page and chunk counts.

PDFs copied manually into `documents/` are auto-discovered and indexed on the next *Process* / *Rebuild*.

> **Scanned / "Print to PDF" files:** PyPDF can only read PDFs that contain a real text layer.
> Image-only PDFs, including files produced by *Microsoft: Print To PDF* from a web page or
> image, have no extractable text and are marked **Failed** ("no extractable text"). Fix: download
> the original text PDF from the department website, or OCR the file first, e.g.
> `ocrmypdf input.pdf output.pdf`, then upload the output. Test: if you cannot select or copy text
> in a PDF viewer, this app cannot read it either.
>
> *Process Documents* only indexes **new** uploads, so a failed file is not re-parsed on every
> click. *Rebuild FAISS Index* retries all documents.

## 5. FAISS persistence

```
vectorstore/
├── document_registry.json     # id, filename, checksum, status, pages, chunks
└── faiss_index/
    ├── index.faiss            # vectors
    ├── index.pkl              # LangChain docstore (pickle)
    └── index_manifest.json    # app signature, embedding model, dimension, chunk ids per doc
```

* **First run:** no index → created on first ingestion and saved.
* **Restart:** the index is loaded from disk at FastAPI startup — **no embeddings are regenerated**.
* **New PDFs:** only the new documents are embedded and appended; the index is re-saved.
* **Delete:** the document's chunk ids (tracked in the manifest) are removed from FAISS, the index is
  re-saved and the PDF is removed from `documents/`. Deleted content is never retrievable afterwards.
* **Rebuild:** all PDFs in `documents/` are re-processed into a brand-new index, which replaces the old
  one only after it is fully written (atomic directory swap).
* **Compatibility:** the manifest stores the embedding model. If `EMBEDDING_MODEL` changes, the old
  index is **refused** (not silently loaded) and the UI asks you to *Rebuild*.
* Writes go to `faiss_index.tmp/` and are swapped in, so a crash never corrupts the live index.

### Security: `allow_dangerous_deserialization`

LangChain saves the FAISS docstore with Python **pickle**. Unpickling a malicious file can execute
arbitrary code, so LangChain requires `allow_dangerous_deserialization=True` to load it. This app:

* passes that flag **only** for its own `vectorstore/faiss_index/` directory, written by this app;
* checks the app-signed `index_manifest.json` before loading (missing/foreign manifest → refused);
* **never** accepts index files via the API or UI.

Do **not** copy FAISS index files from other people or the internet into `vectorstore/`.

## 6. Adding / deleting documents & rebuilding

| Action | UI | API |
|---|---|---|
| Upload | file picker → *Upload & Process* | `POST /documents/upload` (multipart `files`) |
| Index pending | *Upload & Process* | `POST /ingest` |
| List | Documents card | `GET /documents` |
| Delete | 🗑 popover → *Delete* | `DELETE /documents/{document_id}` |
| Rebuild | *Rebuild FAISS Index* | `POST /index/rebuild` |
| Index status | FAISS Index card | `GET /index/status` |
| Health | top banner | `GET /health` |

## 7. Chat

* `POST /chat` `{"question": "...", "history": [...]}` → `{"answer", "sources":[{document, page, relevant_content}]}`
* Retrieval uses the **same** embedding model as indexing; `TOP_K` (default 5) is configurable.
* **Follow-ups:** only the last `HISTORY_TURNS` (default 3) Q/A pairs are sent, truncated. The LLM first
  rewrites the follow-up into a standalone query (e.g. *"How much is it?"* → *"micro irrigation drip
  subsidy amount"*) and that query is used for FAISS retrieval, so answers are always based on the PDFs.
* The system prompt enforces: answer only from context, never invent details, say when information is
  missing, keep documents/years distinct, cite `[filename, p. N]`, don't present old data as current.

## 8. Tests

```bash
pytest -q
```

The suite runs **offline**: it generates real PDFs on the fly, uses a deterministic hashing embedder
and a fake LLM. It covers upload validation/storage, page extraction & metadata, chunking, FAISS
creation/persistence/reload (asserting zero re-embedding), incremental add, delete, rebuild,
incompatible-index refusal, retrieval citations, follow-up condensation and every API endpoint.

Live smoke test against watsonx.ai (requires `.env`):

```bash
python scripts/smoke_llm.py
```

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `WATSONX_TOKEN` | — (required) | watsonx.ai MaaS API token |
| `LLM_BASE_URL` | GPT-OSS-120B MaaS URL | OpenAI-compatible base URL |
| `LLM_MODEL` | `gpt-oss-120b` | Model name |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `LLM_TIMEOUT` | `120` | Seconds per LLM request |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Splitter config |
| `TOP_K` | `5` | Chunks retrieved per question |
| `HISTORY_TURNS` | `3` | Q/A pairs used for follow-ups |
| `MAX_UPLOAD_MB` | `50` | Per-file upload limit |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Used by Streamlit |

## Troubleshooting

* **Document Failed: "no extractable text"**: the PDF is image-only (see section 4). OCR it or use a text PDF.
* `DeprecationWarning: langchain-community is being sunset`: harmless. `PyPDFLoader` and the LangChain
  FAISS integration required by this stack still come from that package.

* **"Cannot reach the backend"** — start Terminal 1 first.
* **"WATSONX_TOKEN is not set"** — fill in `.env` and restart uvicorn.
* **502 from /chat** — watsonx.ai rejected the call (expired token, wrong URL/model, network/VPN).
* **"Index was built with embedding model …"** — you changed `EMBEDDING_MODEL`; click *Rebuild*.
* **Document Failed: no extractable text** — scanned PDF; use a text-based PDF.
