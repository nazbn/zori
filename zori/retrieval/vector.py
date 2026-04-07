from dataclasses import dataclass
from typing import Callable

import chromadb
from chromadb.api.models.Collection import Collection

from zori.ingestion.pdf import TextChunk


@dataclass
class ChunkResult:
    item_key: str
    text: str
    score: float


class ChromaVectorStore:
    def __init__(self, persist_directory: str, embed_fn: Callable[[list[str]], list[list[float]]]):
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embed_fn = embed_fn
        self._collection: Collection = self._client.get_or_create_collection(
            name="zori",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._embed_fn(texts)
        self._collection.add(
            ids=[f"{c.item_key}_{c.chunk_index}" for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"item_key": c.item_key, "chunk_index": c.chunk_index} for c in chunks],
        )

    def delete_item(self, item_key: str) -> None:
        results = self._collection.get(where={"item_key": item_key})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    def search(
        self,
        query: str,
        top_k: int = 5,
        item_keys: list[str] | None = None,
    ) -> list[ChunkResult]:
        query_embedding = self._embed_fn([query])[0]
        where = {"item_key": {"$in": item_keys}} if item_keys else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )

        chunks = []
        for text, distance, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            chunks.append(ChunkResult(
                item_key=meta["item_key"],
                text=text,
                score=1 - distance,  # cosine distance → similarity
            ))
        return chunks


def create_vector_store(
    persist_directory: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> ChromaVectorStore:
    """Instantiate a ChromaVectorStore backed by the given directory and embedding function."""
    return ChromaVectorStore(persist_directory, embed_fn)
