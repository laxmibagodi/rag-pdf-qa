# 🧠 PDF Brain — RAG PDF Q&A App

A production-ready **Retrieval-Augmented Generation (RAG)** app that lets you upload
any PDF(s) and ask questions about them in a conversational chat interface.

Built with **LangChain · FAISS · OpenAI · Streamlit**.

---

## Architecture

```
PDF Upload
   │
   ▼
PyPDFLoader  ──► RecursiveCharacterTextSplitter  ──► Chunks
                                                        │
                                                        ▼
                                              OpenAI Embeddings
                                                        │
                                                        ▼
                                                 FAISS VectorStore
                                                        │
                         ┌──────────────────────────────┘
                         ▼
User Question ──► MMR Retriever (top-k chunks) ──► GPT-3.5-Turbo ──► Answer
```

---

## Quickstart (Local)

### 1. Clone / download the project
```bash
git clone <your-repo>
cd rag_pdf_qa
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your key
```

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and file `app.py`.
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Click **Deploy**. Done! 🎉

---

## File Structure

```
rag_pdf_qa/
├── app.py              ← Streamlit UI (sidebar, chat interface)
├── pdf_processor.py    ← PDF loading, chunking, FAISS indexing
├── rag_chain.py        ← LangChain RAG chain with memory
├── requirements.txt
├── .env.example
└── .streamlit/
    └── secrets.toml.example
```

---

## Key Concepts

| Concept | What it does |
|---------|-------------|
| **PyPDFLoader** | Extracts text from PDFs page by page |
| **RecursiveCharacterTextSplitter** | Splits text into overlapping chunks |
| **OpenAI Embeddings** | Converts text chunks to vectors |
| **FAISS** | Stores vectors; retrieves similar chunks at query time |
| **MMR Retrieval** | Picks *diverse* chunks (avoids duplicate context) |
| **ConversationalRetrievalChain** | Maintains chat history + calls LLM |
| **GPT-3.5-Turbo** | Generates the final answer from retrieved context |

---

## Customisation Ideas

- Swap `text-embedding-3-small` → `text-embedding-3-large` for better accuracy
- Swap `gpt-3.5-turbo` → `gpt-4o` for deeper reasoning
- Add **Chroma** as a persistent vector store (survives restarts)
- Add **HuggingFace embeddings** to run fully offline (free)
- Export chat history as PDF/markdown

---

## Cost Estimate

- Embedding 100 pages of PDF ≈ **~$0.002** (text-embedding-3-small)
- Each Q&A turn ≈ **~$0.001** (gpt-3.5-turbo)

Essentially free for personal use. ✅
