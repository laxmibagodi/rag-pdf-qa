from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import pipeline
import streamlit as st


@st.cache_resource
def load_qa_pipeline():
    """
    deepset/roberta-base-squad2 is a TRUE extractive Q&A model.
    It reads context and extracts the exact answer span — no hallucination.
    """
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
    )


def build_rag_chain(vectorstore, top_k=4):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    return retriever  # we handle the chain manually in ask_question


def ask_question(chain_tuple, question, chat_history):
    # chain_tuple is just the retriever here
    retriever = chain_tuple

    # Step 1: Retrieve relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        return {"answer": "I couldn't find that in the uploaded documents.", "sources": []}

    # Step 2: Combine chunks into one context string
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 3: Run extractive Q&A
    qa_pipeline = load_qa_pipeline()

    try:
        result = qa_pipeline(question=question, context=context, max_answer_len=200)
        answer = result["answer"].strip()
        score = result["score"]

        # If model is not confident, say so
        if score < 0.05 or not answer:
            answer = "I couldn't find a clear answer in the uploaded documents."
    except Exception as e:
        answer = f"Error during answering: {str(e)}"

    # Step 4: Collect sources
    sources = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"{src} (p.{int(page)+1})" if page != "" else src
        if label not in seen:
            sources.append(label)
            seen.add(label)

    return {"answer": answer, "sources": sources}
