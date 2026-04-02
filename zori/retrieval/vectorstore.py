import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chromadb
from chromadb.api.models.Collection import Collection

from zori.ingestion.pdf import TextChunk
from zori.ingestion.zotero import ZoteroItem

METADATA_PATH = Path(".zori/metadata.json")


# ---------------------------------------------------------------------------
# Metadata store
# ---------------------------------------------------------------------------

class MetadataStore:
    """Persists paper-level metadata keyed by Zotero item key."""

    def __init__(self, path: Path = METADATA_PATH):
        self._path = path
        self._data: dict[str, dict] = self._load()

    def save(self, item: ZoteroItem) -> None:
        # Store all PDF attachment keys. v1 uses the first for download.
        # TODO: smarter attachment selection when multiple PDFs exist (future release)
        attachment_keys = [a.key for a in item.attachments]
        self._data[item.key] = {
            "title": item.title,
            "authors": item.authors,
            "year": item.year,
            "journal": item.journal,
            "doi": item.doi,
            "tags": item.tags,
            "item_type": item.item_type,
            "attachment_keys": attachment_keys,
        }
        self._persist()

    def get(self, item_key: str) -> dict | None:
        return self._data.get(item_key)

    def get_attachment_key(self, item_key: str) -> str | None:
        """Return the first attachment key for download. Multi-PDF selection is a future improvement."""
        meta = self._data.get(item_key)
        if not meta:
            return None
        keys = meta.get("attachment_keys", [])
        return keys[0] if keys else None

    def delete(self, item_key: str) -> None:
        self._data.pop(item_key, None)
        self._persist()

    def filter(
        self,
        year: str | None = None,
        tags: list[str] | None = None,
        authors: list[str] | None = None,
    ) -> list[str]:
        """Return item keys matching all provided criteria."""
        results = []
        for key, meta in self._data.items():
            if year and meta.get("year", "")[:4] != year:
                continue
            if tags and not any(t in meta.get("tags", []) for t in tags):
                continue
            if authors and not any(
                a.lower() in " ".join(meta.get("authors", [])).lower()
                for a in authors
            ):
                continue
            results.append(key)
        return results

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {}

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

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
    return ChromaVectorStore(persist_directory, embed_fn)
