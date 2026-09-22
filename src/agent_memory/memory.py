from functools import lru_cache

from sentence_transformers import SentenceTransformer

from agent_memory import db

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def add_semantic_memory(customer_id: int, content: str) -> dict:
    embedding = embed_text(content)
    return db.add_semantic_memory(customer_id, content, embedding)


def retrieve_semantic_memories(customer_id: int, query: str, k: int = 3) -> list[str]:
    query_embedding = embed_text(query)
    rows = db.search_semantic_memories(customer_id, query_embedding, k)
    return [row["content"] for row in rows]


def add_episodic_memory(customer_id: int, order_id: int | None, event: str, outcome: str) -> dict:
    embedding = embed_text(f"{event}. {outcome}.")
    return db.add_episodic_memory(customer_id, order_id, event, outcome, embedding)


def retrieve_episodic_memories(customer_id: int, query: str, k: int = 3) -> list[dict]:
    query_embedding = embed_text(query)
    rows = db.search_episodic_memories(customer_id, query_embedding, k)
    return [{"event": row["event"], "outcome": row["outcome"]} for row in rows]
