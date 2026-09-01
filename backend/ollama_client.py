"""
Thin wrapper around the local Ollama HTTP API.
Assumes Ollama is running locally at http://localhost:11434 (its default).
"""

import time
import requests

OLLAMA_BASE_URL = "http://localhost:11434"

# Model used to generate answers (you said you have this pulled)
CHAT_MODEL = "llama3.1"

# Model used to generate embeddings for RAG.
# "nomic-embed-text" is the standard lightweight embedding model for Ollama.
# Pull it once with:  ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    """Get a single embedding vector from Ollama for the given text."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Ollama's /api/embed endpoint supports a single input at a time,
    so we loop here. For a couple of small PDFs this is fine speed-wise.
    """
    return [get_embedding(t) for t in texts]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Ask llama3.1 to answer STRICTLY using the provided context chunks.
    If the context doesn't contain the answer, the model is instructed
    to say so explicitly using a fixed marker string we can detect.

    Retries up to 3 times on server errors (Ollama may 500 while
    cold-loading the model into memory).
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are a support assistant that answers questions ONLY using the "
        "context provided below, which comes from the company's Privacy "
        "Policy and Terms & Conditions documents.\n\n"
        "Rules:\n"
        "1. Only use information present in the CONTEXT section. Do not use "
        "any outside knowledge.\n"
        "2. If the CONTEXT does not contain enough information to answer the "
        "question, respond with EXACTLY this phrase and nothing else: "
        "NOT_FOUND_IN_DOCS\n"
        "3. Do not guess, speculate, or fill gaps with general knowledge.\n"
        "4. Keep answers concise and directly reference the relevant policy "
        "point.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    return _chat_with_retry(system_prompt, question)


def generate_hybrid_answer(
    question: str,
    context_chunks: list[str],
    web_context: str,
) -> str:
    """
    Hybrid mode: answer using BOTH internal policy documents AND web search
    results. The model is told to prefer internal docs and clearly label
    web-sourced information.
    """
    doc_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(none)"

    system_prompt = (
        "You are a knowledgeable assistant. You have two sources of information:\n\n"
        "1. INTERNAL DOCUMENTS (from the company's Privacy Policy and Terms "
        "& Conditions):\n"
        f"{doc_context}\n\n"
        "2. WEB SEARCH RESULTS:\n"
        f"{web_context or '(none)'}\n\n"
        "Rules:\n"
        "- Prefer internal document information when it is relevant.\n"
        "- Supplement with web search results when the internal docs are "
        "insufficient.\n"
        "- Clearly indicate when information comes from web search.\n"
        "- Keep answers concise and well-structured.\n"
    )

    return _chat_with_retry(system_prompt, question)


def generate_standalone_answer(question: str) -> str:
    """
    Standalone LLM mode: answer directly from the model's training data,
    with no retrieval or external context.
    """
    system_prompt = (
        "You are a helpful, concise, and knowledgeable AI assistant. "
        "Answer the user's question to the best of your ability. "
        "Be accurate and well-structured in your responses."
    )

    return _chat_with_retry(system_prompt, question)


# ── internal helper ──────────────────────────────────────────────

def _chat_with_retry(system_prompt: str, user_message: str) -> str:
    """Send a chat request to Ollama with retry logic for cold-start 500s."""
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180,
        )
        if response.status_code == 500 and attempt < max_retries:
            time.sleep(2 * attempt)
            continue
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


