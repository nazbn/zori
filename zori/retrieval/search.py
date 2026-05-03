import logging
from dataclasses import dataclass

from langchain_classic.retrievers import EnsembleRetriever

from zori.retrieval.metadata import MetadataStore
from zori.retrieval.retrievers import (
    ChunksRetriever,
    MetadataRetriever,
    PapersRetriever,
    TagsRetriever,
    TitleRetriever,
)
from zori.retrieval.vector import ZoriVectorStore

logger = logging.getLogger(__name__)


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
    def __init__(
        self,
        vector_store: ZoriVectorStore,
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
        """Hybrid search: all signals combined via EnsembleRetriever (equal-weight RRF).

        author and year are soft scoring signals (via MetadataRetriever), not hard filters.
        tags score via TagsRetriever. title triggers high-precision FTS5 on the title column.
        Results are deduplicated by item_key — multiple chunks from the same paper collapse.
        """
        retrievers = []

        if tags and self._lexical_index:
            retrievers.append(TagsRetriever(lexical_index=self._lexical_index, tags=tags,
                                            metadata={"query": " ".join(tags)}))

        if title and self._lexical_index:
            retrievers.append(TitleRetriever(lexical_index=self._lexical_index, title=title,
                                             metadata={"query": title}))

        if lexical_queries and self._lexical_index:
            for lq in lexical_queries:
                retrievers.append(PapersRetriever(lexical_index=self._lexical_index, query=lq,
                                                  metadata={"query": lq}))
                retrievers.append(ChunksRetriever(lexical_index=self._lexical_index, query=lq,
                                                  metadata={"query": lq}))

        if semantic_query:
            retrievers.append(self._vector_store.as_retriever(
                search_kwargs={"k": top_k * 2},
                metadata={"query": semantic_query},
            ))

        if author or year:
            retrievers.append(MetadataRetriever(
                metadata_store=self._metadata_store,
                author=author,
                year=year,
                metadata={"query": " ".join(filter(None, [author, year]))},
            ))

        if not retrievers:
            return []

        # id_key="item_key" deduplicates across retrievers by metadata["item_key"],
        # so multiple chunks from the same paper collapse into one result.
        ensemble = EnsembleRetriever(retrievers=retrievers, id_key="item_key")
        docs = ensemble.invoke(semantic_query or "")

        results = []
        for rank, doc in enumerate(docs[:top_k]):
            key = doc.metadata["item_key"]
            meta = self._metadata_store.get(key) or {}
            results.append(SearchResult(
                item_key=key,
                text="",
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                journal=meta.get("journal"),
                score=1.0 / (60 + rank + 1),
            ))

        return results
