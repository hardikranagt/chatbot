"""
Handles querying Chroma and deciding whether the retrieved chunks are
actually relevant enough to answer from (closed-domain gate).
"""

import os
import chromadb

from gemini_client import get_embedding

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "company_policies"

# Cosine DISTANCE threshold (Chroma returns distance, lower = more similar).
# 0.0 = identical, ~1.0+ = unrelated. Tune this after testing with your docs.
# Start conservative; loosen (increase) if relevant questions get rejected,
# tighten (decrease) if irrelevant questions get answered.
RELEVANCE_DISTANCE_THRESHOLD = 0.75

TOP_K = 4


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve_relevant_chunks(question: str):
    """
    Returns (chunks, metadatas, is_relevant).
    is_relevant is False if nothing retrieved clears the similarity threshold
    -- this is what keeps the bot from answering off-topic questions using
    whatever happens to be the "least bad" match in the DB.
    """
    collection = get_collection()

    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not documents:
        return [], [], False

    # Keep only chunks that pass the relevance bar
    relevant_docs = []
    relevant_meta = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist <= RELEVANCE_DISTANCE_THRESHOLD:
            relevant_docs.append(doc)
            relevant_meta.append({**meta, "distance": dist})

    is_relevant = len(relevant_docs) > 0
    return relevant_docs, relevant_meta, is_relevant
