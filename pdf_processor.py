import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def process_pdfs(uploaded_files, chunk_size=512, chunk_overlap=50):
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            for doc in docs:
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

    # 🔥 FREE embeddings (no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Create FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)