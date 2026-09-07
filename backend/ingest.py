"""
Ingestion pipeline for the Privacy Policy + Terms & Conditions Markdown files.

Run this ONCE (or whenever the .md files change) to (re)build the Chroma
vector store:

    python ingest.py

It expects two files inside ./data/:
    data/privacy_policy.md
    data/terms_and_conditions.md

You can rename these via the FILES list below if your filenames differ.
"""

import os
import uuid

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from gemini_client import get_embeddings_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "company_policies"

# Map: (filename in data/, doc_type label stored in metadata)
FILES = [
    ("privacy_policy.md", "privacy_policy"),
    ("terms_and_conditions.md", "terms_and_conditions"),
]

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 120
BATCH_SIZE = 20        # how many chunks to embed per Gemini API call batch


def extract_sections(md_path: str):
    """
    Read a Markdown file and split it into logical sections by top-level (#)
    or second-level (##) headings. Each section is returned as a (section_num,
    text) tuple — analogous to (page_num, text) in the old PDF reader.
    Falls back to treating the whole file as a single section if no headings
    are found.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        raw = f.read()

    sections = []
    current_lines = []
    section_num = 1

    for line in raw.splitlines(keepends=True):
        # Split on H1 or H2 headings
        if line.startswith("# ") or line.startswith("## "):
            if current_lines:
                text = "".join(current_lines).strip()
                if text:
                    sections.append((section_num, text))
                    section_num += 1
                current_lines = []
        current_lines.append(line)

    # Flush remaining lines
    if current_lines:
        text = "".join(current_lines).strip()
        if text:
            sections.append((section_num, text))

    if not sections:
        # No headings — treat whole file as section 1
        sections = [(1, raw.strip())]

    return sections


def chunk_sections(sections, source_label: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for section_num, text in sections:
        for idx, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "page": section_num,   # kept as "page" for metadata compatibility
                "source": source_label,
                "chunk_index_on_page": idx,
            })
    return chunks


def build_index():
    db = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh rebuild each time ingest.py is run, so stale chunks never linger
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = db.create_collection(
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
        sections = extract_sections(path)
        print(f"  Extracted {len(sections)} sections")
        chunks = chunk_sections(sections, doc_type)
        print(f"  Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks created. Check that your .md files are in backend/data/.")
        return

    print(f"\nEmbedding {len(all_chunks)} total chunks via Gemini "
          f"(this calls the Gemini embedding API)...")

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
