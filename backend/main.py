from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

from retriever import retrieve_relevant_chunks
from ollama_client import (
    generate_answer,
    generate_hybrid_answer,
    generate_standalone_answer,
)
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


class ChatRequest(BaseModel):
    question: str
    mode: Literal["rag", "hybrid", "standalone"] = "rag"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Standalone LLM ──────────────────────────────────────────
    if req.mode == "standalone":
        try:
            answer = generate_standalone_answer(question)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed. Is Ollama running? Error: {e}",
            )
        return ChatResponse(answer=answer, sources=[], mode="standalone")

    # ── Retrieval (shared by RAG and Hybrid) ────────────────────
    try:
        chunks, metadatas, is_relevant = retrieve_relevant_chunks(question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Retrieval failed. Make sure Ollama is running locally and "
                f"you've run ingest.py to build the vector store. Error: {e}"
            ),
        )

    sources = [
        {
            "document": m["source"],
            "page": m["page"],
            "relevance_distance": round(m["distance"], 4),
        }
        for m in metadatas
    ] if metadatas else []

    # ── RAG mode ────────────────────────────────────────────────
    if req.mode == "rag":
        if not is_relevant:
            return ChatResponse(answer=NOT_FOUND_MESSAGE, sources=[], mode="rag")

        try:
            answer = generate_answer(question, chunks)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed. Is Ollama running? Error: {e}",
            )

        if "NOT_FOUND_IN_DOCS" in answer:
            return ChatResponse(answer=NOT_FOUND_MESSAGE, sources=[], mode="rag")

        return ChatResponse(answer=answer, sources=sources, mode="rag")

    # ── Hybrid mode (RAG + Web Search) ──────────────────────────
    if req.mode == "hybrid":
        web_results = search_web(question)
        web_context = format_web_results(web_results)

        web_sources = [
            {"document": r["title"], "link": r["link"], "type": "web"}
            for r in web_results
        ]

        try:
            answer = generate_hybrid_answer(question, chunks, web_context)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed. Is Ollama running? Error: {e}",
            )

        all_sources = sources + web_sources
        return ChatResponse(answer=answer, sources=all_sources, mode="hybrid")

    raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")
