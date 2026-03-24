import tempfile
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader  # 🔥 better fallback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 🔥 CLEANING FUNCTION (FIXES SPACED TEXT ISSUE)
def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove weird spacing like: "S w a s t i k"
    if len(text) > 0 and text.count(" ") > len(text) * 0.3:
        text = text.replace(" ", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_pdf_with_fallback(path):
    """
    Try PyPDFLoader first, fallback to PyMuPDFLoader if text looks broken.
    """
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Check if text is garbage (too many spaced characters)
    sample = " ".join([doc.page_content[:200] for doc in docs])
    if sample.count(" ") > len(sample) * 0.3:
        # 🔥 fallback
        loader = PyMuPDFLoader(path)
        docs = loader.load()

    return docs


def process_pdfs(uploaded_files, chunk_size=512, chunk_overlap=50):
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            docs = load_pdf_with_fallback(tmp_path)

            for doc in docs:
                # 🔥 CLEAN TEXT HERE
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = uploaded_file.name

            all_docs.extend(docs)

        finally:
            os.unlink(tmp_path)

    if not all_docs:
        raise ValueError("No text could be extracted.")

    # 🔹 Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_docs)

    # 🔥 FREE embeddings (stable + fast)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)
