import streamlit as st
from dotenv import load_dotenv
from pdf_processor import process_pdfs
from rag_chain import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional Theme + Sidebar Fix ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --primary: #2563eb;
    --primary-soft: #2563eb22;
    --bg: #f8fafc;
    --surface: #ffffff;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
}

/* Base */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    padding: 1rem 1rem 2rem 1rem !important;
    overflow-y: auto !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem !important;
}

section[data-testid="stSidebar"] > div {
    overflow: visible !important;
}

/* Buttons */
.stButton > button {
    width: 100% !important;
    background: var(--primary);
    color: white;
    border-radius: 6px;
    border: none;
    font-weight: 500;
}
.stButton > button:hover {
    background: #1d4ed8;
}

/* Inputs */
.stTextInput input {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
}
.stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px var(--primary-soft);
}

/* File uploader */
[data-testid="stFileUploader"] {
    width: 100% !important;
    padding: 0.5rem !important;
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* Sliders spacing */
.stSlider {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

/* Chat */
.chat-user {
    background: #eef2ff;
    padding: 10px;
    border-radius: 10px;
    margin: 8px 0;
}
.chat-bot {
    background: white;
    border-left: 3px solid var(--primary);
    padding: 10px;
    border-radius: 6px;
    margin: 8px 0;
}

/* Cards */
.stat-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--primary);
}

/* Source chips */
.source-chip {
    display: inline-block;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 16px;
    padding: 3px 10px;
    font-size: 0.7rem;
    color: #3730a3;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "num_docs" not in st.session_state:
    st.session_state.num_docs = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0'>
            <div style='font-size:2rem'>📄</div>
            <div style='font-size:1.2rem; font-weight:600'>Document Intelligence</div>
            <div style='color:#6b7280; font-size:0.8rem'>Ingest • Analyze • Query</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("### Settings")

    chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64)
    overlap = st.slider("Chunk Overlap", 0, 200, 50, 25)
    top_k = st.slider("Top-K Retrieval", 1, 8, 4)

    st.markdown("")

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing documents..."):
                vectorstore, n_chunks = process_pdfs(uploaded_files, chunk_size, overlap)
                st.session_state.vectorstore = vectorstore
                st.session_state.num_docs = len(uploaded_files)
                st.session_state.num_chunks = n_chunks
                st.session_state.chat_history = []
            st.success(f"{len(uploaded_files)} document(s) processed successfully.")

    if st.session_state.vectorstore:
        st.divider()
        col1, col2 = st.columns(2)
        col1.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.num_docs}</div><div>Documents</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.num_chunks}</div><div>Chunks</div></div>', unsafe_allow_html=True)

    st.divider()

    if st.button("Clear Data"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.num_docs = 0
        st.session_state.num_chunks = 0
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Document Q&A Assistant")
st.caption("Upload documents and interact with them using AI-powered retrieval.")

if not st.session_state.vectorstore:
    st.info("Upload PDFs and process them to begin querying.")
else:
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for turn in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user"><b>You:</b> {turn["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot"><b>Answer:</b> {turn["answer"]}</div>', unsafe_allow_html=True)

            for s in turn.get("sources", []):
                st.markdown(f'<span class="source-chip">{s}</span>', unsafe_allow_html=True)

    with st.form("qa_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        question = col1.text_input("Ask a question", placeholder="e.g. What are the key clauses?")
        submitted = col2.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Generating answer..."):
            retriever = build_rag_chain(st.session_state.vectorstore, top_k)
            result = ask_question(retriever, question, st.session_state.chat_history)

        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

    if not st.session_state.chat_history:
        st.subheader("Try asking")
        suggestions = [
            "Summarize the document",
            "Extract key insights",
            "List important entities",
            "What are the main obligations?",
        ]

        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s):
                with st.spinner("Generating answer..."):
                    retriever = build_rag_chain(st.session_state.vectorstore, top_k)
                    result = ask_question(retriever, s, [])

                st.session_state.chat_history.append({
                    "question": s,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()
