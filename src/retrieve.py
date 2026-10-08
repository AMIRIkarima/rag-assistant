from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def retrieve(question: str, k: int = 5) -> list[tuple[str, dict]]:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    collection = _collection()
    if collection.count() == 0:
        raise RuntimeError("The paper collection is empty. Index papers with `python -m src.ingest` first.")

    query_embedding = _embedding_model().encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(documents, metadatas))