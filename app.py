import streamlit as st
from dotenv import load_dotenv
from pdf_processor import process_pdfs
from rag_chain import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(page_title="PDF Brain — RAG Q&A", page_icon="🌸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Space+Grotesk:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --pink:       #ff2d78;
    --pink-light: #ff6fa8;
    --pink-glow:  #ff2d7833;
    --pink-soft:  #ff2d7811;
    --bg:         #0a0008;
    --surface:    #120010;
    --surface2:   #1c0018;
    --border:     #3a0030;
    --text:       #ffe4f0;
    --muted:      #a06080;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Animated gradient background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, #ff2d7820 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, #c0006020 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--pink) !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #ff2d78, #c0006a) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 0 18px #ff2d7844 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff6fa8, #ff2d78) !important;
    box-shadow: 0 0 30px #ff2d7877 !important;
    transform: translateY(-2px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--pink) !important;
    box-shadow: 0 0 0 2px var(--pink-glow) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* ── Chat bubbles ── */
.chat-user {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 3px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    position: relative;
}
.chat-bot {
    background: var(--surface);
    border-left: 3px solid var(--pink);
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    line-height: 1.7;
    font-size: 0.95rem;
    box-shadow: -4px 0 20px var(--pink-glow);
}

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: var(--pink-soft);
    border: 1px solid var(--pink);
    border-radius: 20px;
    padding: 3px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--pink-light);
    margin: 4px 4px 4px 0;
    letter-spacing: 0.03em;
}

/* ── Stat cards ── */
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 0 20px var(--pink-soft);
}
.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: var(--pink);
    text-shadow: 0 0 20px var(--pink-glow);
}

/* ── Info / success / warning ── */
.stAlert { border-radius: 8px !important; }
[data-testid="stNotification"] { border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: var(--pink) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--pink); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "num_docs" not in st.session_state:
    st.session_state.num_docs = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem'>
            <div style='font-size:2.5rem'>🌸</div>
            <div style='font-family:Playfair Display,serif; font-size:1.5rem; color:#ff6fa8'>PDF Brain</div>
            <div style='color:#a06080; font-size:0.8rem; margin-top:4px'>Upload · Embed · Ask</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded_files = st.file_uploader("Drop your PDFs here", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64)
    overlap = st.slider("Chunk Overlap", 0, 200, 50, 25)
    top_k = st.slider("Top-K Chunks", 1, 8, 4)

    st.markdown("")
    if st.button("✨ Process PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Chunking & embedding…"):
                vectorstore, n_chunks = process_pdfs(uploaded_files, chunk_size, overlap)
                st.session_state.vectorstore = vectorstore
                st.session_state.num_docs = len(uploaded_files)
                st.session_state.num_chunks = n_chunks
                st.session_state.chat_history = []
            st.success(f"✅ {len(uploaded_files)} PDF(s) — {n_chunks} chunks ready!")

    if st.session_state.vectorstore:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.num_docs}</div><div style="font-size:0.72rem;color:#a06080;margin-top:4px">PDFs</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.num_chunks}</div><div style="font-size:0.72rem;color:#a06080;margin-top:4px">Chunks</div></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑 Clear All", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.num_docs = 0
        st.session_state.num_chunks = 0
        st.rerun()

    st.markdown('<div style="color:#3a0030;font-size:0.68rem;text-align:center;margin-top:2rem">Powered by HuggingFace · FAISS · LangChain</div>', unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='margin-bottom: 0.25rem'>
        <span style='font-family:Playfair Display,serif; font-size:3rem; font-weight:700'>Ask Your</span>
        <span style='font-family:Playfair Display,serif; font-size:3rem; font-style:italic; color:#ff2d78'> Documents</span>
    </div>
    <p style='color:#a06080; margin-bottom:2rem'>Upload PDFs on the left, then ask anything below.</p>
""", unsafe_allow_html=True)

if not st.session_state.vectorstore:
    st.info("🌸 Upload PDFs in the sidebar and click **Process PDFs** to get started.")
else:
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for turn in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">🙋 {turn["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">🌸 {turn["answer"]}</div>', unsafe_allow_html=True)
            for s in turn.get("sources", []):
                st.markdown(f'<span class="source-chip">📄 {s}</span>', unsafe_allow_html=True)
        st.markdown("")

    with st.form("qa_form", clear_on_submit=True):
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            question = st.text_input("Ask a question…", placeholder="e.g. What is the main conclusion?", label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Ask →")

    if submitted and question.strip():
        with st.spinner("Thinking…"):
            retriever = build_rag_chain(st.session_state.vectorstore, top_k)
            result = ask_question(retriever, question, st.session_state.chat_history)
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

    if not st.session_state.chat_history:
        st.markdown("#### ✨ Try asking…")
        suggestions = [
            "Summarise this document.",
            "What are the key findings?",
            "List all dates mentioned.",
            "Who are the main people involved?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}"):
                with st.spinner("Thinking…"):
                    retriever = build_rag_chain(st.session_state.vectorstore, top_k)
                    result = ask_question(retriever, s, [])
                st.session_state.chat_history.append({
                    "question": s,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()
