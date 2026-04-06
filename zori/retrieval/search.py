from dataclasses import dataclass

from zori.retrieval.metadata import MetadataStore
from zori.retrieval.vector import ChunkResult, ChromaVectorStore


@dataclass
class SearchResult:
    text: str
    item_key: str
    title: str
    authors: list[str]
    year: str | None
    journal: str | None
    score: float


def _rrf_combine(ranked_lists: list[list[str]], weights: list[float], k: int = 60) -> list[str]:
    """Weighted Reciprocal Rank Fusion. Returns item_keys ordered by combined RRF score."""
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_key in enumerate(ranked):
            scores[item_key] = scores.get(item_key, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


class SearchService:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        metadata_store: MetadataStore,
        lexical_index=None,  # LexicalIndex | None — avoid circular import
    ):
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._lexical_index = lexical_index

    def hybrid_search(
        self,
        lexical_queries: list[str] | None = None,
        semantic_query: str | None = None,
        title: str | None = None,
        author: str | None = None,
        year: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Always-hybrid search: lexical + semantic run in parallel, combined via RRF.

        author and year act as hard pre-filters; tags score via RRF not as a filter.
        `title` triggers high-precision FTS5 scoped to the title column.
        """
        # 1. Metadata pre-filter (author and year only — tags score via RRF)
        filter_keys: set[str] | None = None
        if author or year:
            keys = self._metadata_store.filter(
                year=year,
                authors=[author] if author else None,
            )
            filter_keys = set(keys)

        # 2. Collect ranked lists + weights + track vector chunk text
        ranked_lists: list[list[str]] = []
        weights: list[float] = []
        vector_chunks: dict[str, ChunkResult] = {}

        # Tag-scoped BM25
        if tags and self._lexical_index:
            tag_hits = [k for k, _ in self._lexical_index.search_tags(tags)]
            if tag_hits:
                ranked_lists.append(tag_hits)
                weights.append(3.0)

        # Title-scoped BM25 (high precision for specific paper/acronym lookup)
        if title and self._lexical_index:
            title_keys = self._lexical_index.search_title(title)
            if title_keys:
                ranked_lists.append(title_keys)
                weights.append(3.0)

        # Lexical BM25 on papers_fts and chunks_fts — one ranked list per query
        if lexical_queries and self._lexical_index:
            for lq in lexical_queries:
                paper_hits = [k for k, _ in self._lexical_index.search_papers(lq)]
                if paper_hits:
                    ranked_lists.append(paper_hits)
                    weights.append(3.0)
                chunk_hits = [k for k, _ in self._lexical_index.search_chunks(lq)]
                if chunk_hits:
                    ranked_lists.append(chunk_hits)
                    weights.append(2.0)

        # Vector (semantic) search
        if semantic_query:
            vec_filter = list(filter_keys) if filter_keys else None
            try:
                chunks = self._vector_store.search(
                    semantic_query, top_k=top_k * 2, item_keys=vec_filter
                )
                vec_keys: list[str] = []
                for c in chunks:
                    vec_keys.append(c.item_key)
                    if c.item_key not in vector_chunks or c.score > vector_chunks[c.item_key].score:
                        vector_chunks[c.item_key] = c
                if vec_keys:
                    ranked_lists.append(vec_keys)
                    weights.append(1.0)
            except Exception:
                pass

        if not ranked_lists:
            return []

        # 3. Weighted RRF combine
        combined = _rrf_combine(ranked_lists, weights)

        # 4. Apply metadata filter (lexical results may include out-of-filter papers)
        if filter_keys is not None:
            combined = [k for k in combined if k in filter_keys]

        # 5. Build SearchResult list
        results: list[SearchResult] = []
        for rank, key in enumerate(combined[:top_k]):
            meta = self._metadata_store.get(key) or {}
            chunk = vector_chunks.get(key)
            rrf_score = 1.0 / (60 + rank + 1)
            results.append(SearchResult(
                item_key=key,
                text=chunk.text if chunk else "",
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                journal=meta.get("journal"),
                score=chunk.score if chunk else rrf_score,
            ))

        return results

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        item_keys: list[str] | None = None,
    ) -> list[SearchResult]:
        """Semantic search over chunk embeddings."""
        chunks = self._vector_store.search(query, top_k=top_k, item_keys=item_keys)
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
