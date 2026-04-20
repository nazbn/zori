from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from zori.ingestion.pdf import TextChunk


class ZoriVectorStore:
    def __init__(self, persist_directory: str, embeddings: Embeddings):
        self._store = Chroma(
            collection_name="zori",
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        self._store.add_texts(
            texts=[c.text for c in chunks],
            ids=[f"{c.item_key}_{c.chunk_index}" for c in chunks],
            metadatas=[{"item_key": c.item_key, "chunk_index": c.chunk_index} for c in chunks],
        )

    def delete_item(self, item_key: str) -> None:
        results = self._store.get(where={"item_key": item_key})
        if results["ids"]:
            self._store.delete(ids=results["ids"])

    def as_retriever(self, **kwargs):
        return self._store.as_retriever(**kwargs)


def create_vector_store(persist_directory: str, embeddings: Embeddings) -> ZoriVectorStore:
    """Instantiate a ZoriVectorStore backed by LangChain Chroma."""
    return ZoriVectorStore(persist_directory, embeddings)
