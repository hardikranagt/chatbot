import json
from collections.abc import Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal

from retriever import retrieve_relevant_chunks
import gemini_client
import ollama_client
from web_search import search_web, format_web_results

app = FastAPI(title="Policy RAG Chatbot")

# Allow the React dev server to call this API.
# Tighten allow_origins to your actual frontend URL in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOT_FOUND_MESSAGE = (
    "I couldn't find anything relevant to that in our Privacy Policy or "
    "Terms & Conditions. For further help, please contact our support team."
)

# Maximum number of past messages to send to the LLM for context.
# 10 messages ≈ 5 user-assistant exchanges — keeps prompts bounded.
MAX_HISTORY = 10


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    mode: Literal["rag", "hybrid", "standalone"] = "rag"
    history: list[HistoryMessage] = []
    model_name: str = "gemini-3.6-flash"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Shared retrieval helper ──────────────────────────────────────

def _retrieve(question: str) -> tuple[list[str], list[dict], bool, list[dict]]:
    """Run vector retrieval and return chunks, metadatas, relevance flag, sources."""
    try:
        chunks, metadatas, is_relevant = retrieve_relevant_chunks(question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Retrieval failed. Make sure your GEMINI_API_KEY is set and "
                f"you've run ingest.py to build the vector store. Error: {e}"
            ),
        )
    sources = [
        {
            "document": m["source"],
            "page": m["page"],
            "relevance_distance": round(m["distance"], 4),
        }
        for m in (metadatas or [])
    ]
    return chunks, metadatas, is_relevant, sources


# ── Streaming endpoint ───────────────────────────────────────────

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Server-Sent Events endpoint.
    Streams tokens as they arrive from the LLM so the UI feels instant.

    SSE format:
      data: {"type": "token",   "text": "..."}
      data: {"type": "sources", "sources": [...], "mode": "..."}
      data: {"type": "done"}
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [msg.model_dump() for msg in req.history[-MAX_HISTORY:]]
    use_ollama = (req.model_name == "ollama")

    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def generate() -> Generator[str, None, None]:
        try:
            # ── Standalone ──────────────────────────────────────────
            if req.mode == "standalone":
                if use_ollama:
                    answer = ollama_client.generate_standalone_answer(question, history=history)
                    yield event({"type": "token", "text": answer})
                else:
                    for chunk in gemini_client.stream_standalone_answer(question, history, req.model_name):
                        yield event({"type": "token", "text": chunk})
                yield event({"type": "sources", "sources": [], "mode": "standalone"})
                yield event({"type": "done"})
                return

            # ── Retrieval (RAG + Hybrid) ─────────────────────────────
            chunks, _, is_relevant, sources = _retrieve(question)

            # ── RAG ─────────────────────────────────────────────────
            if req.mode == "rag":
                if not is_relevant:
                    yield event({"type": "token", "text": NOT_FOUND_MESSAGE})
                    yield event({"type": "sources", "sources": [], "mode": "rag"})
                    yield event({"type": "done"})
                    return

                full_text = ""
                if use_ollama:
                    answer = ollama_client.generate_answer(question, chunks, history=history)
                    full_text = answer
                    yield event({"type": "token", "text": answer})
                else:
                    for chunk in gemini_client.stream_answer(question, chunks, history, req.model_name):
                        full_text += chunk
                        yield event({"type": "token", "text": chunk})

                if "NOT_FOUND_IN_DOCS" in full_text:
                    yield event({"type": "replace", "text": NOT_FOUND_MESSAGE})
                    yield event({"type": "sources", "sources": [], "mode": "rag"})
                else:
                    yield event({"type": "sources", "sources": sources, "mode": "rag"})
                yield event({"type": "done"})
                return

            # ── Hybrid ───────────────────────────────────────────────
            if req.mode == "hybrid":
                web_results = search_web(question)
                web_context = format_web_results(web_results)
                web_sources = [
                    {"document": r["title"], "link": r["link"], "type": "web"}
                    for r in web_results
                ]

                if use_ollama:
                    answer = ollama_client.generate_hybrid_answer(question, chunks, web_context, history=history)
                    yield event({"type": "token", "text": answer})
                else:
                    for chunk in gemini_client.stream_hybrid_answer(question, chunks, web_context, history, req.model_name):
                        yield event({"type": "token", "text": chunk})

                yield event({"type": "sources", "sources": sources + web_sources, "mode": "hybrid"})
                yield event({"type": "done"})
                return

            yield event({"type": "error", "message": f"Unknown mode: {req.mode}"})
            yield event({"type": "done"})

        except Exception as e:
            # Always send a clean error + done so the frontend stream never hangs
            yield event({"type": "error", "message": str(e)})
            yield event({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")



# ── Non-streaming endpoint (kept for compatibility) ──────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = [msg.model_dump() for msg in req.history[-MAX_HISTORY:]]

    if req.mode == "standalone":
        try:
            if req.model_name == "ollama":
                answer = ollama_client.generate_standalone_answer(question, history=history)
            else:
                answer = gemini_client.generate_standalone_answer(question, history=history, model_name=req.model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed. Error: {e}")
        return ChatResponse(answer=answer, sources=[], mode="standalone")

    chunks, _, is_relevant, sources = _retrieve(question)

    if req.mode == "rag":
        if not is_relevant:
            return ChatResponse(answer=NOT_FOUND_MESSAGE, sources=[], mode="rag")
        try:
            if req.model_name == "ollama":
                answer = ollama_client.generate_answer(question, chunks, history=history)
            else:
                answer = gemini_client.generate_answer(question, chunks, history=history, model_name=req.model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed. Error: {e}")
        if "NOT_FOUND_IN_DOCS" in answer:
            return ChatResponse(answer=NOT_FOUND_MESSAGE, sources=[], mode="rag")
        return ChatResponse(answer=answer, sources=sources, mode="rag")

    if req.mode == "hybrid":
        web_results = search_web(question)
        web_context = format_web_results(web_results)
        web_sources = [
            {"document": r["title"], "link": r["link"], "type": "web"}
            for r in web_results
        ]
        try:
            if req.model_name == "ollama":
                answer = ollama_client.generate_hybrid_answer(question, chunks, web_context, history=history)
            else:
                answer = gemini_client.generate_hybrid_answer(question, chunks, web_context, history=history, model_name=req.model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed. Error: {e}")
        return ChatResponse(answer=answer, sources=sources + web_sources, mode="hybrid")

    raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
