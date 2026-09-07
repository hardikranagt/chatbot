"""
Gemini API client for embeddings and LLM generation.
Uses the new `google-genai` SDK (replaces the deprecated google-generativeai).

Requires GEMINI_API_KEY in the .env file.
"""

import os
import time
from collections.abc import Generator

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it to backend/.env"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Models ──────────────────────────────────────────────────────
CHAT_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-001"


# ── Embeddings ──────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Get a single embedding vector from Gemini for the given text."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for a batch of texts.
    The new SDK supports list input natively.
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [emb.values for emb in result.embeddings]


# ── Helpers ─────────────────────────────────────────────────────

def _build_gemini_history(history: list[dict] | None) -> list[types.Content]:
    """Convert frontend history to Gemini Content objects."""
    if not history:
        return []
    result = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        result.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            )
        )
    return result


def _make_config(system_prompt: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
    )


# ── Non-streaming generation (full response at once) ────────────

def _chat_with_retry(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    config = _make_config(system_prompt)
    gemini_history = _build_gemini_history(history)

    FALLBACK_MODEL = "gemini-3.5-flash"
    models_to_try = [model_name]
    if model_name != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_exc = None
    for model in models_to_try:
        for attempt in range(1, 4):   # up to 3 retries per model
            try:
                chat = client.chats.create(
                    model=model,
                    config=config,
                    history=gemini_history,
                )
                response = chat.send_message(user_message)
                return response.text.strip()
            except Exception as e:
                last_exc = e
                err_str = str(e).upper()
                is_overloaded = "503" in err_str or "UNAVAILABLE" in err_str
                is_not_found = "404" in err_str or "NOT_FOUND" in err_str
                if is_not_found:
                    break  # model doesn't exist, skip to fallback immediately
                if is_overloaded and attempt < 3:
                    time.sleep(2 ** attempt)  # 2s, 4s
                    continue
                break  # non-retryable error or retries exhausted

    raise last_exc


# ── Streaming generation (token-by-token) ───────────────────────

def stream_chat(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> Generator[str, None, None]:
    """
    Yields text chunks as they arrive from Gemini.
    Retries on 503/UNAVAILABLE up to 3 times with exponential backoff,
    then falls back to gemini-1.5-flash before giving up.
    """
    config = _make_config(system_prompt)
    gemini_history = _build_gemini_history(history)

    FALLBACK_MODEL = "gemini-3.5-flash"
    models_to_try = [model_name]
    if model_name != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_exc = None
    for model in models_to_try:
        for attempt in range(1, 4):
            try:
                chat = client.chats.create(
                    model=model,
                    config=config,
                    history=gemini_history,
                )
                for chunk in chat.send_message_stream(user_message):
                    if chunk.text:
                        yield chunk.text
                return  # success — exit generator
            except Exception as e:
                last_exc = e
                err_str = str(e).upper()
                is_overloaded = "503" in err_str or "UNAVAILABLE" in err_str
                is_not_found = "404" in err_str or "NOT_FOUND" in err_str
                if is_not_found:
                    break  # model doesn't exist, try fallback
                if is_overloaded and attempt < 3:
                    time.sleep(2 ** attempt)  # 2s, 4s
                    continue
                break  # non-retryable or retries exhausted

    raise last_exc


# ── RAG mode ────────────────────────────────────────────────────

def _rag_system_prompt(context_chunks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return (
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
        f"point.\n\nCONTEXT:\n{context_text}"
    )


def generate_answer(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    return _chat_with_retry(_rag_system_prompt(context_chunks), question, history, model_name)


def stream_answer(
    question: str,
    context_chunks: list[str],
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> Generator[str, None, None]:
    return stream_chat(_rag_system_prompt(context_chunks), question, history, model_name)


# ── Hybrid mode ─────────────────────────────────────────────────

def _hybrid_system_prompt(context_chunks: list[str], web_context: str) -> str:
    doc_context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(none)"
    return (
        "You are a knowledgeable assistant. You have two sources of information:\n\n"
        "1. INTERNAL DOCUMENTS (from the company's Privacy Policy and Terms "
        f"& Conditions):\n{doc_context}\n\n"
        "2. WEB SEARCH RESULTS:\n"
        f"{web_context or '(none)'}\n\n"
        "Rules:\n"
        "- Prefer internal document information when it is relevant.\n"
        "- Supplement with web search results when the internal docs are insufficient.\n"
        "- Clearly indicate when information comes from web search.\n"
        "- Keep answers concise and well-structured.\n"
    )


def generate_hybrid_answer(
    question: str,
    context_chunks: list[str],
    web_context: str,
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    return _chat_with_retry(_hybrid_system_prompt(context_chunks, web_context), question, history, model_name)


def stream_hybrid_answer(
    question: str,
    context_chunks: list[str],
    web_context: str,
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> Generator[str, None, None]:
    return stream_chat(_hybrid_system_prompt(context_chunks, web_context), question, history, model_name)


# ── Standalone mode ─────────────────────────────────────────────

def _standalone_system_prompt() -> str:
    return (
        "You are a helpful, concise, and knowledgeable AI assistant. "
        "Answer the user's question to the best of your ability. "
        "Be accurate and well-structured in your responses."
    )


def generate_standalone_answer(
    question: str,
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> str:
    return _chat_with_retry(_standalone_system_prompt(), question, history, model_name)


def stream_standalone_answer(
    question: str,
    history: list[dict] | None = None,
    model_name: str = "gemini-3.6-flash",
) -> Generator[str, None, None]:
    return stream_chat(_standalone_system_prompt(), question, history, model_name)
