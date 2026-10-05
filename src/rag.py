from __future__ import annotations

import re
from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "irrigation_guidance.md"


def load_chunks() -> list[str]:
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", text)
    return [section.strip() for section in sections if section.strip()]


def retrieve(question: str, chunks: list[str], count: int = 3) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    indexes = scores.argsort()[::-1][:count]
    return [chunks[index] for index in indexes if scores[index] > 0]


def ask_ollama(question: str, context: str, field_summary: str, model: str = DEFAULT_MODEL) -> str:
    prompt = f"""You are a local sugarcane irrigation assistant.
Answer only questions about sugarcane irrigation and the current field conditions.
Use only the supplied context. If the context does not support an answer, say so.
Do not invent irrigation duration, water quantity, sensor readings, or medical facts.
Do not discuss diseases, fertilizer, yield, or unrelated agriculture.
Mention that the ML result uses synthetic data when relevant.

Current field prediction:
{field_summary}

Retrieved irrigation context:
{context}

User question: {question}
"""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def ollama_is_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False