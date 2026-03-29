from dataclasses import dataclass

from zori.retrieval.vectorstore import ChromaVectorStore, MetadataStore


@dataclass
class SearchResult:
    text: str
    item_key: str
    title: str
    authors: list[str]
    year: str | None
    journal: str | None
    score: float


class SearchService:
    def __init__(self, vector_store: ChromaVectorStore, metadata_store: MetadataStore):
        self._vector_store = vector_store
        self._metadata_store = metadata_store

    def search(
        self,
        query: str,
        top_k: int = 5,
        year: str | None = None,
        tags: list[str] | None = None,
        authors: list[str] | None = None,
    ) -> list[SearchResult]:
        item_keys = None
        if any([year, tags, authors]):
            item_keys = self._metadata_store.filter(year=year, tags=tags, authors=authors)
            if not item_keys:
                return []

        chunks = self._vector_store.search(query, top_k=top_k, item_keys=item_keys)

        results = []
        for chunk in chunks:
            meta = self._metadata_store.get(chunk.item_key) or {}
            results.append(SearchResult(
                text=chunk.text,
                item_key=chunk.item_key,
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                journal=meta.get("journal"),
                score=chunk.score,
            ))
        return results
