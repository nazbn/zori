from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict


class PapersRetriever(BaseRetriever):
    """BM25 retriever over papers_fts (title, abstract, tags)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lexical_index: Any
    query: str
    top_k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.lexical_index.search_papers(self.query, top_k=self.top_k)
        return [Document(page_content=key, metadata={"item_key": key}) for key, _ in results]


class ChunksRetriever(BaseRetriever):
    """BM25 retriever over chunks_fts (full paper text)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lexical_index: Any
    query: str
    top_k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.lexical_index.search_chunks(self.query, top_k=self.top_k)
        return [Document(page_content=key, metadata={"item_key": key}) for key, _ in results]


class TagsRetriever(BaseRetriever):
    """BM25 retriever scoped to the tags column."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lexical_index: Any
    tags: list[str]
    top_k: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = self.lexical_index.search_tags(self.tags, top_k=self.top_k)
        return [Document(page_content=key, metadata={"item_key": key}) for key, _ in results]


class TitleRetriever(BaseRetriever):
    """High-precision FTS5 retriever scoped to the title column."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lexical_index: Any
    title: str
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        keys = self.lexical_index.search_title(self.title, top_k=self.top_k)
        return [Document(page_content=key, metadata={"item_key": key}) for key in keys]


class MetadataRetriever(BaseRetriever):
    """Retriever that boosts papers matching author and/or year."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata_store: Any
    author: str | None = None
    year: str | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        keys = self.metadata_store.filter(
            year=self.year,
            authors=[self.author] if self.author else None,
        )
        return [Document(page_content=key, metadata={"item_key": key}) for key in keys]
