# Policy RAG Chatbot — Backend

## Prerequisites

1. **Ollama installed and running locally** (you said it already is).
2. Pull the two models this project uses:

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

   `llama3.1` generates answers. `nomic-embed-text` is Ollama's standard
   embedding model, used to turn text into vectors for Chroma.

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Add your documents

Place your two PDFs in `backend/data/`, named exactly:

```
data/privacy_policy.pdf
data/terms_and_conditions.pdf
```

(If your filenames differ, edit the `FILES` list at the top of `ingest.py`.)

## Build the vector index

Run this once, and again any time the PDFs change:

```bash
python ingest.py
```

This extracts text from both PDFs, chunks it, embeds each chunk via your
local Ollama `nomic-embed-text` model, and stores everything in a local
Chroma database at `backend/chroma_db/`.

## Run the API server

```bash
uvicorn main:app --reload --port 8000
```

Test it directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How long do you retain my personal data?"}'
```

Ask something unrelated (e.g. "what's the weather today?") and you should
get the "couldn't find anything relevant" fallback instead of a hallucinated
answer — that's the closed-domain behavior working as intended.

## How the "only answer from these 2 files" restriction works

Two layers:

1. **Similarity gate** (`retriever.py`): the user's question is embedded and
   compared against stored chunks. If nothing scores within
   `RELEVANCE_DISTANCE_THRESHOLD`, the backend returns the fallback message
   *without even calling the LLM*.
2. **Prompt-level instruction** (`ollama_client.py`): even when relevant
   chunks ARE retrieved, the system prompt forces llama3.1 to answer only
   from that context and to emit `NOT_FOUND_IN_DOCS` if it genuinely can't
   answer from what was retrieved. The backend catches that marker and
   converts it to the same friendly fallback message.

## Tuning tips

- If **relevant** questions are getting rejected: increase
  `RELEVANCE_DISTANCE_THRESHOLD` in `retriever.py` (e.g. 0.75 → 0.9).
- If **off-topic** questions are getting answered: decrease it
  (e.g. 0.75 → 0.6).
- If chunks feel too fragmented or too broad, adjust `CHUNK_SIZE` /
  `CHUNK_OVERLAP` in `ingest.py` and re-run ingestion.
- `TOP_K` in `retriever.py` controls how many chunks get passed to the LLM
  as context per question.
