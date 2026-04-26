from langchain_chroma import Chroma
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

    def add_chunks(self, chunks: list[TextChunk], batch_size: int = 50) -> None:
        if not chunks:
            return
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._store.add_texts(
                texts=[c.text for c in batch],
                ids=[f"{c.item_key}_{c.chunk_index}" for c in batch],
                metadatas=[{"item_key": c.item_key, "chunk_index": c.chunk_index} for c in batch],
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
