
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG Application",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Application")
st.write(
    "Upload multiple PDF documents and ask questions "
    "across all your documents."
)


# --------------------------------------------------
# 2. Initialize Session State
# --------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# --------------------------------------------------
# 3. Initialize Embedding Model
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 4. Initialize IBM GPT-OSS-120B
# --------------------------------------------------

@st.cache_resource
def get_llm():
    token = os.getenv("WATSONX_TOKEN")
    base_url = os.getenv("WATSONX_BASE_URL")
    model_name = os.getenv("WATSONX_MODEL_NAME")

    if not all([token, base_url, model_name]):
        raise ValueError(
            "Missing IBM watsonx credentials. "
            "Check your .env file."
        )

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=token,
        temperature=0
    )


# --------------------------------------------------
# 5. Upload Multiple PDFs
# --------------------------------------------------

st.header("Step 1: Upload Documents")

files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if files:
    st.write("Uploaded documents:")

    for file in files:
        st.write(f"📄 {file.name}")


# --------------------------------------------------
# 6. Split and Index All Documents
# --------------------------------------------------

st.header("Step 2: Build Knowledge Base")

if st.button("Process and Index PDFs", type="primary"):

    if not files:
        st.warning("Please upload at least one PDF.")
    else:

        all_documents = []
        all_chunks = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        progress = st.progress(0)

        try:
            for index, file in enumerate(files):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(file.getvalue())
                    temp_path = temp_file.name

                try:
                    # Load PDF pages
                    loader = PyPDFLoader(temp_path)
                    documents = loader.load()

                    # Preserve original filename
                    for document in documents:
                        document.metadata["source"] = file.name

                    all_documents.extend(documents)

                    # Split into chunks
                    chunks = splitter.split_documents(documents)

                    all_chunks.extend(chunks)

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                progress.progress(
                    (index + 1) / len(files)
                )

            if not all_chunks:
                st.error(
                    "No readable text was found in the PDFs."
                )
            else:

                # Create ONE vector database
                embeddings = get_embeddings()

                st.session_state.store = FAISS.from_documents(
                    documents=all_chunks,
                    embedding=embeddings
                )

                st.session_state.chunk_count = len(all_chunks)

                st.session_state.uploaded_names = [
                    file.name for file in files
                ]

                st.success(
                    f"Successfully indexed "
                    f"{len(all_chunks)} chunks!"
                )

                st.write(
                    f"Documents processed: {len(files)}"
                )

                st.write(
                    f"Total chunks indexed: {len(all_chunks)}"
                )

        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")


# --------------------------------------------------
# 7. Display Knowledge Base Information
# --------------------------------------------------

if st.session_state.store is not None:

    st.divider()

    st.subheader("Knowledge Base Status")

    col1, col2 = st.columns(2)

    col1.metric(
        "PDF Documents",
        len(st.session_state.uploaded_names)
    )

    col2.metric(
        "Chunks Indexed",
        st.session_state.chunk_count
    )

    st.write("Indexed files:")

    for name in st.session_state.uploaded_names:
        st.write(f"📄 {name}")


# --------------------------------------------------
# 8. Ask Questions Across All PDFs
# --------------------------------------------------

st.divider()
st.header("Step 3: Ask Questions")

question = st.text_input(
    "Enter your question",
    placeholder="Ask something about your uploaded PDFs..."
)

if st.button("Get Answer"):

    if st.session_state.store is None:
        st.warning("Please index your PDFs first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        try:
            # Retrieve relevant chunks from all PDFs
            retriever = st.session_state.store.as_retriever(
                search_kwargs={"k": 5}
            )

            docs = retriever.invoke(question)

            if not docs:
                st.warning(
                    "No relevant information was found."
                )
                st.stop()

            # Prepare context with source information
            context_parts = []

            for doc in docs:

                source = doc.metadata.get(
                    "source", "Unknown"
                )

                # PyPDFLoader page numbers start from zero
                page = doc.metadata.get("page")

                page_number = (
                    int(page) + 1 if page is not None
                    else "Unknown"
                )

                context_parts.append(
                    f"""
                    Source: {source}
                    Page: {page_number}

                    Content:
                    {doc.page_content}
                    """
                )

            context = "\n\n".join(context_parts)

            # Construct prompt
            prompt = ChatPromptTemplate.from_template(
                """
                You are a document question-answering assistant.

                Answer the user's question using ONLY the
                information provided in the retrieved context.

                Rules:
                1. Do not invent information.
                2. Cite sources using the filename and page number.
                3. If documents disagree, clearly explain the
                   disagreement and cite both sources.
                4. If information is insufficient, say so.
                5. Do not silently choose one conflicting answer.

                Retrieved Context:
                {context}

                User Question:
                {question}

                Provide a clear and concise answer with citations.
                """
            )

            # Connect prompt to IBM LLM
            chain = prompt | get_llm()

            response = chain.invoke({
                "context": context,
                "question": question
            })

            # Display answer
            st.subheader("Answer")

            st.write(response.content)

            # Display source references
            st.subheader("Sources")

            seen_sources = set()

            for doc in docs:

                source = doc.metadata.get(
                    "source", "Unknown"
                )

                page = doc.metadata.get("page")

                page_number = (
                    int(page) + 1 if page is not None
                    else "Unknown"
                )

                source_key = (source, page_number)

                if source_key not in seen_sources:

                    st.caption(
                        f"📄 {source} - Page {page_number}"
                    )

                    seen_sources.add(source_key)

        except Exception as e:
            st.error(f"Question answering failed: {str(e)}")
