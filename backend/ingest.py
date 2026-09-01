"""
Ingestion pipeline for the Privacy Policy + Terms & Conditions PDFs.

Run this ONCE (or whenever the PDFs change) to (re)build the Chroma
vector store:

    python ingest.py

It expects two files inside ./data/:
    data/privacy_policy.pdf
    data/terms_and_conditions.pdf

You can rename these via the FILES list below if your filenames differ.
"""

import os
import uuid

import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ollama_client import get_embeddings_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "company_policies"

# Map: (filename in data/, doc_type label stored in metadata)
FILES = [
    ("privacy_policy.pdf", "privacy_policy"),
    ("terms_and_conditions.pdf", "terms_and_conditions"),
]

CHUNK_SIZE = 800       # characters (kept simple/offline-safe, no tokenizer download needed)
CHUNK_OVERLAP = 120
BATCH_SIZE = 20        # how many chunks to embed per Ollama call batch


def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_pages(pages, source_label: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page_num, text in pages:
        for idx, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "page": page_num,
                "source": source_label,
                "chunk_index_on_page": idx,
            })
    return chunks


def build_index():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh rebuild each time ingest.py is run, so stale chunks never linger
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    for filename, doc_type in FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue
        print(f"Processing {filename}...")
        pages = extract_pages(path)
        print(f"  Extracted {len(pages)} pages")
        chunks = chunk_pages(pages, doc_type)
        print(f"  Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks created. Check that your PDFs are in backend/data/.")
        return

    print(f"\nEmbedding {len(all_chunks)} total chunks via Ollama "
          f"(this calls your local Ollama server)...")

    for start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[start:start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = get_embeddings_batch(texts)

        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index_on_page": c["chunk_index_on_page"],
                }
                for c in batch
            ],
        )
        done = min(start + BATCH_SIZE, len(all_chunks))
        print(f"  {done}/{len(all_chunks)} embedded and stored")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")


if __name__ == "__main__":
    build_index()
