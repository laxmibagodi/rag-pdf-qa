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

# ── Enterprise Professional Theme ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.cdnfonts.com/css/sf-pro-display');

:root {
    --primary: #1e40af;
    --primary-dark: #1e3a8a;
    --primary-light: #3b82f6;
    --secondary: #64748b;
    --success: #059669;
    --bg-primary: #f8fafc;
    --bg-secondary: #f1f5f9;
    --surface: #ffffff;
    --surface-alt: #f8fafc;
    --border: #e2e8f0;
    --border-hover: #cbd5e1;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --radius: 12px;
}

/* Base Reset */
* {
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #e2e8f0 100%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
    font-weight: 400;
    line-height: 1.6;
}

/* Main Content */
[data-testid="stAppViewContainer"] > .main {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar - Professional Panel */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    padding: 2rem 1.5rem 2rem 1.5rem !important;
    box-shadow: var(--shadow-lg);
    min-width: 320px !important;
}

[data-testid="stSidebar"] .css-1d391kg {
    padding-top: 0 !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem !important;
}

.st-emotion-cache-1r1jky0 {
    font-size: 1.875rem !important;
    font-weight: 700 !important;
}

/* Buttons - Professional */
.stButton > button {
    width: 100% !important;
    height: 48px !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.2s ease !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg) !important;
    background: linear-gradient(135deg, var(--primary-dark) 0%, #1e40af 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary Buttons */
button[kind="secondary"] {
    background: var(--surface-alt) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

button[kind="secondary"]:hover {
    background: var(--bg-secondary) !important;
    border-color: var(--border-hover) !important;
}

/* Inputs - Clean Professional */
.stTextInput > div > div > input {
    height: 48px !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0 1rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.1) !important;
    outline: none !important;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface-alt) !important;
    padding: 2rem !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background: linear-gradient(135deg, var(--surface-alt) 0%, rgba(30, 64, 175, 0.02) 100%) !important;
}

/* Sliders */
.stSlider > div > div > div {
    height: 6px !important;
    border-radius: 3px !important;
}

/* Cards & Stats */
.metric-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    text-align: center !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
    height: 100%;
}

.metric-card:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px) !important;
}

.metric-number {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    color: var(--primary) !important;
    line-height: 1.2 !important;
}

.metric-label {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    margin-top: 0.25rem !important;
}

/* Chat Messages */
.message-user {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: var(--radius) !important;
    padding: 1.25rem 1.5rem !important;
    margin: 1rem 0 !important;
}

.message-assistant {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 4px solid var(--primary) !important;
    border-radius: var(--radius) !important;
    padding: 1.25rem 1.5rem !important;
    margin: 1rem 0 !important;
    box-shadow: var(--shadow-sm) !important;
}

.source-chip {
    display: inline-flex !important;
    align-items: center !important;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 20px !important;
    padding: 0.25rem 0.75rem !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: var(--primary) !important;
    margin: 0.25rem !important;
    text-decoration: none !important;
}

/* Status Messages */
.stAlert {
    border-radius: var(--radius) !important;
    border: none !important;
    padding: 1rem 1.25rem !important;
}

/* Divider */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 2rem 0 !important;
}

/* Form */
.stForm {
    background: var(--surface-alt) !important;
    border-radius: var(--radius) !important;
    padding: 2rem !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}
::-webkit-scrollbar-thumb {
    background: var(--border-hover);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--secondary);
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
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 0 0 2rem 0'>
            <div style='font-size: 3rem; line-height: 1'>📄</div>
            <h2 style='margin: 0.5rem 0 0.25rem 0; font-size: 1.5rem; font-weight: 700; color: var(--text-primary)'>
                Document Intelligence
            </h2>
            <div style='color: var(--text-muted); font-size: 0.85rem; font-weight: 400'>
                AI-Powered Document Analysis
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # File Upload
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDFs to analyze"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Settings
    st.subheader("⚙️ Processing Settings")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size", 256, 1024, 512, 64, 
                              help="Size of text chunks for embedding")
    with col2:
        overlap = st.slider("Overlap", 0, 200, 50, 25,
                           help="Overlap between chunks")
    
    top_k = st.slider("Top-K Results", 1, 8, 4, 1,
                     help="Number of document chunks to retrieve")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Process Button
    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("👆 Please upload at least one PDF first.")
        else:
            with st.spinner("🔄 Processing your documents..."):
                vectorstore, n_chunks = process_pdfs(uploaded_files, chunk_size, overlap)
                st.session_state.vectorstore = vectorstore
                st.session_state.num_docs = len(uploaded_files)
                st.session_state.num_chunks = n_chunks
                st.session_state.chat_history = []
            st.success(f"✅ {len(uploaded_files)} document(s) processed successfully!")

    # Stats
    if st.session_state.vectorstore:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{st.session_state.num_docs}</div>
                    <div class="metric-label">Documents</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{st.session_state.num_chunks:,}</div>
                    <div class="metric-label">Text Chunks</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.session_state.num_docs = 0
            st.session_state.num_chunks = 0
            st.rerun()

# ── Main Content ──────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    # Welcome Screen
    st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem'>
            <div style='font-size: 5rem; margin-bottom: 2rem'>📄</div>
            <h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text'>
                Document Intelligence
            </h1>
            <p style='font-size: 1.25rem; color: var(--text-secondary); max-width: 600px; margin: 0 auto 3rem'>
                Upload your PDF documents and unlock powerful AI-driven insights, 
                summaries, and answers instantly.
            </p>
            <div style='font-size: 1rem; color: var(--text-muted); margin-bottom: 3rem'>
                🔒 Secure • ⚡ Fast • 🎯 Accurate
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 **Step 1:** Upload PDFs in the sidebar")
else:
    # Chat Interface
    st.markdown("<h2 style='margin-bottom: 1rem'>💬 Ask Questions</h2>", unsafe_allow_html=True)
    st.caption("Interact with your documents using natural language")

    # Chat History
    if st.session_state.chat_history:
        st.markdown("### 📜 Conversation History")
        for i, turn in enumerate(st.session_state.chat_history):
            with st.container():
                col1, col2 = st.columns([1, 0.1])
                with col1:
                    st.markdown(f"""
                        <div class="message-user">
                            <strong>👤 You:</strong><br>
                            <span style='color: var(--text-primary)'>{turn["question"]}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="message-assistant">
                            <strong>🤖 AI:</strong><br>
                            <span style='color: var(--text-primary); line-height: 1.6'>{turn["answer"]}</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("📋", key=f"copy_{i}"):
                        st.success("Copied to clipboard!")
            
            # Sources
            if turn.get("sources"):
                sources_html = " ".join([f'<span class="source-chip">{s}</span>' for s in turn["sources"]])
                st.markdown(f'<div style="margin-top: 0.5rem">{sources_html}</div>', unsafe_allow_html=True)

    # Question Form
    with st.form("question_form", clear_on_submit=True):
        st.markdown("### ➤ New Question")
        col1, col2 = st.columns([5, 1])
        question = col1.text_input(
            "", 
            placeholder="e.g. What are the key obligations? Summarize section 3? ...",
            label_visibility="collapsed"
        )
        submitted = col2.form_submit_button("➤ Ask", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("🤖 Generating response..."):
            retriever = build_rag_chain(st.session_state.vectorstore, top_k)
            result = ask_question(retriever, question, st.session_state.chat_history)

        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })
        st.rerun()

    # Quick Questions
    if not st.session_state.chat_history:
        st.markdown("### ✨ Try These Questions")
        suggestions = [
            "📋 Summarize the entire document",
            "🔍 Extract all key entities and dates", 
            "⚖️ What are the main obligations?",
            "📊 Provide key insights and statistics"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, use_container_width=True):
                with st.spinner("🤖 Generating response..."):
                    retriever = build_rag_chain(st.session_state.vectorstore, top_k)
                    result = ask_question(retriever, suggestion, [])
                
                st.session_state.chat_history.append({
                    "question": suggestion,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()
