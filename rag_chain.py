import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


@st.cache_resource
def load_qa_model():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_answer_from_context(question: str, context: str) -> dict:
    tokenizer, model = load_qa_model()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        padding=True,
    )

    best_answer = ""
    best_score = -float("inf")

    for i in range(inputs["input_ids"].shape[0]):
        input_ids = inputs["input_ids"][i].unsqueeze(0)
        attention_mask = inputs["attention_mask"][i].unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        start_scores = outputs.start_logits[0]
        end_scores = outputs.end_logits[0]

        start_idx = torch.argmax(start_scores).item()
        end_idx = torch.argmax(end_scores).item()
        score = (start_scores[start_idx] + end_scores[end_idx]).item()

        if end_idx >= start_idx and score > best_score:
            best_score = score
            tokens = inputs["input_ids"][i][start_idx: end_idx + 1]
            best_answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

    return {"answer": best_answer, "score": best_score}


def build_rag_chain(vectorstore, top_k=4):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 3},
    )
    return retriever


def ask_question(retriever, question: str, chat_history):
    # Step 1: Retrieve relevant chunks
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "I couldn't find that in the uploaded documents.", "sources": []}

    best_answer = ""
    best_score = -float("inf")

    # Step 2a: Try each chunk individually
    for doc in docs:
        context = doc.page_content.strip()
        if not context:
            continue
        result = get_answer_from_context(question, context)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    # Step 2b: Also try combined context
    # Helps when the answer spans across chunk boundaries
    combined_context = " ".join(
        doc.page_content.strip() for doc in docs if doc.page_content.strip()
    )
    if combined_context:
        result = get_answer_from_context(question, combined_context)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    # Step 3: Confidence check
    # RoBERTa returns raw logits, NOT probabilities (range ~-10 to +10).
    # A threshold of 0.5 was incorrectly rejecting valid answers.
    # -5 is a safe lower bound for a real answer being present.
    if not best_answer or best_score < -5:
        best_answer = "I couldn't find a clear answer in the uploaded documents."

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

    return {"answer": best_answer, "sources": sources}
