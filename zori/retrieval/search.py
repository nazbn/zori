from dataclasses import dataclass

from zori.retrieval.vectorstore import ChunkResult, ChromaVectorStore, MetadataStore


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

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        item_keys: list[str] | None = None,
    ) -> list[SearchResult]:
        """Semantic search over chunk embeddings."""
        chunks = self._vector_store.search(query, top_k=top_k, item_keys=item_keys)
        return [self._to_result(c) for c in chunks]

    def title_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Match against paper titles, then re-rank with vector search."""
        keys = self._metadata_store.title_search(query)
        if not keys:
            return []
        chunks = self._vector_store.search(query, top_k=top_k, item_keys=keys)
        return [self._to_result(c) for c in chunks]

    def author_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Filter by author name, then vector search within those papers."""
        keys = self._metadata_store.author_search(query)
        if not keys:
            return []
        chunks = self._vector_store.search(query, top_k=top_k, item_keys=keys)
        return [self._to_result(c) for c in chunks]

    def tag_search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Filter by tag, then vector search within those papers."""
        keys = self._metadata_store.tag_search(query)
        if not keys:
            return []
        chunks = self._vector_store.search(query, top_k=top_k, item_keys=keys)
        return [self._to_result(c) for c in chunks]

    # kept for the CLI one-shot `zori search` command
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Simple vector search used by the standalone search command."""
        return self.vector_search(query, top_k=top_k)

    def _to_result(self, chunk: ChunkResult) -> SearchResult:
        meta = self._metadata_store.get(chunk.item_key) or {}
        return SearchResult(
            text=chunk.text,
            item_key=chunk.item_key,
            title=meta.get("title", "Unknown"),
            authors=meta.get("authors", []),
            year=meta.get("year"),
            journal=meta.get("journal"),
            score=chunk.score,
        )
