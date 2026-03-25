# 🧠 PDF Brain — RAG PDF Q&A App

A production-ready **Retrieval-Augmented Generation (RAG)** app that lets you upload
any PDF(s) and ask questions about them in a conversational chat interface.

---

## 🚀 Tech Stack

- **Streamlit** — Interactive UI  
- **LangChain** — Document processing + retrieval  
- **FAISS** — Vector database  
- **HuggingFace Transformers** — QA model  
- **Sentence Transformers** — Embeddings  
- **PyPDFLoader / PyMuPDF** — PDF parsing  

---

## 🧠 Architecture


PDF Upload
│
▼
PyPDFLoader / PyMuPDF (fallback)
│
▼
Text Cleaning + Preprocessing
│
▼
RecursiveCharacterTextSplitter
│
▼
Text Chunks
│
▼
HuggingFace Embeddings (MiniLM)
│
▼
FAISS Vector Store
│
▼
MMR Retriever (Top-K)
│
▼
Routing Logic
├── Extractive QA → RoBERTa SQuAD2
└── Open-ended → Chunk-based Summary
│
▼
Final Answer + Sources


---

## ✨ Key Features

- 📄 Upload and process multiple PDFs  
- ⚡ Fully **offline-capable** (no OpenAI required)  
- 💬 Conversational Q&A interface  
- 🔍 Semantic search using embeddings  
- 🎯 MMR retrieval (diverse + relevant chunks)  
- 🧠 Smart routing:
  - Extractive QA (precise answers)
- 🧾 Source attribution (file + page number)  
- 🛠 Adjustable chunking parameters  

---

## 🔥 Unique Highlights

### 🧠 Hybrid QA System
- Uses **RoBERTa SQuAD2** for precise answer extraction  
- Falls back to **context stitching** for summarization  
- Detects intent using keyword-based routing  

### 🧹 Robust PDF Handling
- Automatically switches between:
  - `PyPDFLoader`
  - `PyMuPDFLoader` (fallback)
- Cleans noisy text (e.g., spaced characters)

### 💸 Zero API Cost
- No OpenAI dependency  
- Runs locally with HuggingFace models  

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/laxmibagodi/rag-pdf-qa.git
cd rag-pdf-qa
2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Run the app
streamlit run app.py

Open in browser:

http://localhost:8501
📁 Project Structure
rag-pdf-qa/
├── app.py              # Streamlit UI (chat interface)
├── pdf_processor.py    # PDF loading, cleaning, chunking, embeddings
├── rag_chain.py        # Retrieval + QA + routing logic
├── requirements.txt
└── README.md
🔑 Core Components
Component	Description
PyPDFLoader / PyMuPDF	Extracts text from PDFs
Text Cleaning	Fixes broken spacing and formatting
Text Splitter	Splits into overlapping chunks
MiniLM Embeddings	Converts text → vectors
FAISS	Fast similarity search
MMR Retriever	Improves diversity of results
RoBERTa SQuAD2	Extracts exact answers
Routing Logic	Chooses QA vs summary mode
