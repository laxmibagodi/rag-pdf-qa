from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

ffrom langchain_huggingface import HuggingFaceEndpoint
import streamlit as st


_QA_TEMPLATE = """You are a helpful assistant that answers questions based only on the provided context.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."
Be concise but complete.

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

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        temperature=0.5,
        huggingfacehub_api_token=st.secrets["HUGGINGFACE_API_KEY"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(chain_tuple, question, chat_history):
    chain, retriever = chain_tuple
    answer = chain.invoke(question)

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
