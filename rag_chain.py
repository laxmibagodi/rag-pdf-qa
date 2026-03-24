from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

import streamlit as st


# 🔥 IMPROVED PROMPT (STRICT + CLEAN OUTPUT)
_QA_TEMPLATE = """Answer the question using ONLY the context below.

If the answer is not in the context, say:
"I couldn't find that in the uploaded documents."

Give a clean, structured answer (use bullet points if needed).

Context:
{context}

Question: {question}

Answer:"""

_QA_PROMPT = PromptTemplate.from_template(_QA_TEMPLATE)


def build_rag_chain(vectorstore, top_k=4):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    pipe = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    max_new_tokens=200,
    temperature=0.2,
)

    llm = HuggingFacePipeline(pipeline=pipe)

    # 🔥 LIMIT CONTEXT SIZE (prevents overflow + bad answers)
    def format_docs(docs):
        return "\n\n".join(doc.page_content[:500] for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def clean_answer(text: str) -> str:
    """
    Removes prompt leakage and junk repetition
    """
    if not text:
        return ""

    # Remove accidental prompt echoes
    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    # Remove excessive repetition
    lines = text.split("\n")
    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip()
        if line and line not in seen:
            cleaned.append(line)
            seen.add(line)

    return "\n".join(cleaned).strip()


def ask_question(chain_tuple, question, chat_history):
    chain, retriever = chain_tuple

    raw_answer = chain.invoke(question)
    answer = clean_answer(raw_answer)

    docs = retriever.invoke(question)

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
