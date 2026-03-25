import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import re

# ── Keywords that signal the user wants a summary or listing, not a QA span ──
SUMMARY_TRIGGERS = [
    "summarise", "summarize", "summary", "overview", "brief",
    "list all", "list the", "what are", "who are", "mention all",
    "key points", "key findings", "main points", "tell me about",
    "describe", "explain", "what is this document", "what does this document",
]


def is_open_ended(question: str) -> bool:
    q = question.lower().strip()
    return any(trigger in q for trigger in SUMMARY_TRIGGERS)


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


def make_summary_answer(docs) -> str:
    """
    For open-ended / summarisation queries, stitch the top chunks
    together into a readable passage instead of extracting a span.
    """
    parts = []
    for doc in docs:
        text = doc.page_content.strip()
        if text:
            text = re.sub(r"\n{3,}", "\n\n", text)
            parts.append(text)

    if not parts:
        return "I couldn't find enough content in the uploaded documents."

    combined = "\n\n".join(parts)

    # Trim to a readable length (~1200 chars)
    if len(combined) > 1200:
        combined = combined[:1200].rsplit(" ", 1)[0] + "…"

    return combined


def ask_question(retriever, question: str, chat_history):
    # Step 1: Retrieve relevant chunks
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "I couldn't find that in the uploaded documents.", "sources": []}

    # Step 2: Route — summarisation vs extractive QA
    if is_open_ended(question):
        # For open-ended questions, return the relevant chunks directly
        # roberta-base-squad2 is extractive only — it cannot summarise
        answer = make_summary_answer(docs)
    else:
        best_answer = ""
        best_score = -float("inf")

        # Try each chunk individually
        for doc in docs:
            context = doc.page_content.strip()
            if not context:
                continue
            result = get_answer_from_context(question, context)
            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]

        # Also try combined context (catches answers near chunk boundaries)
        combined_context = " ".join(
            doc.page_content.strip() for doc in docs if doc.page_content.strip()
        )
        if combined_context:
            result = get_answer_from_context(question, combined_context)
            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]

        # Confidence check — RoBERTa returns raw logits, NOT probabilities
        if not best_answer or best_score < -5:
            best_answer = "I couldn't find a clear answer in the uploaded documents."

        answer = best_answer

    # Step 3: Collect sources
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
