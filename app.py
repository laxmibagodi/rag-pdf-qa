import rag_chain
print("Loaded from:", rag_chain.__file__)
import streamlit as st
from dotenv import load_dotenv
from pdf_processor import process_pdfs
from rag_chain import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(page_title="PDF Brain — RAG Q&A", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
:root { --accent: #e8ff47; --bg: #0d0d0d; --surface: #161616; --border: #2a2a2a; --text: #f0ece4; --muted: #888; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }
.stButton > button { background: var(--accent) !important; color: #0d0d0d !important; border: none !important; border-radius: 4px !important; font-family: 'DM Mono', monospace !important; font-weight: 500 !important; }
.stButton > button:hover { background: #fff !important; }
.stTextInput > div > div > input { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 4px !important; }
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; }
.chat-user { background: #1a1a1a; border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0; }
.chat-bot { background: var(--surface); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; margin: 0.5rem 0; line-height: 1.65; }
.source-chip { display: inline-block; background: #1e1e1e; border: 1px solid var(--border); border-radius: 20px; padding: 2px 12px; font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--accent); margin: 4px 4px 4px 0; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }
.stat-number { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "num_docs" not in st.session_state:
    st.session_state.num_docs = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 PDF Brain")
    st.markdown("*Upload → Embed → Ask*")
    st.divider()

    uploaded_files = st.file_uploader("Drop your PDFs here", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64)
    overlap = st.slider("Chunk Overlap", 0, 200, 50, 25)
    top_k = st.slider("Top-K Chunks", 1, 8, 4)

    if st.button("⚡ Process PDFs", use_container_width=True):
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
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.num_docs}</div><div style="font-size:0.75rem;color:#888">PDFs</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.num_chunks}</div><div style="font-size:0.75rem;color:#888">Chunks</div></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑 Clear All", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.num_docs = 0
        st.session_state.num_chunks = 0
        st.rerun()

# Main
st.markdown("# Ask Your *Documents*")
st.markdown('<p style="color:#888">Upload PDFs on the left, then ask anything below.</p>', unsafe_allow_html=True)

if not st.session_state.vectorstore:
    st.info("👈 Upload PDFs in the sidebar and click **Process PDFs** to get started.")
else:
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for turn in st.session_state.chat_history:
            st.markdown(f'<div class="chat-user">🙋 {turn["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bot">🧠 {turn["answer"]}</div>', unsafe_allow_html=True)
            for s in turn.get("sources", []):
                st.markdown(f'<span class="source-chip">📄 {s}</span>', unsafe_allow_html=True)

    with st.form("qa_form", clear_on_submit=True):
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            question = st.text_input("Ask a question…", placeholder="e.g. What is the main conclusion?", label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("Ask →")

    if submitted and question.strip():
        with st.spinner("Thinking…"):
            chain_tuple = build_rag_chain(st.session_state.vectorstore, top_k)
            result = ask_question(chain_tuple, question, st.session_state.chat_history)
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

    if not st.session_state.chat_history:
        st.markdown("#### Try asking…")
        suggestions = ["Summarise this document.", "What are the key findings?", "List all dates mentioned.", "Who are the main people involved?"]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sug_{i}"):
                with st.spinner("Thinking…"):
                    chain_tuple = build_rag_chain(st.session_state.vectorstore, top_k)
                    result = ask_question(chain_tuple, s, [])
                st.session_state.chat_history.append({"question": s, "answer": result["answer"], "sources": result["sources"]})
                st.rerun()