from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
import os


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float


@dataclass
class EmbeddingsConfig:
    provider: str
    model: str


@dataclass
class VectorStoreConfig:
    provider: str
    persist_directory: str
    qdrant_url: str | None


@dataclass
class IngestionConfig:
    mode: str
    sync_on_startup: bool
    chunk_size: int
    chunk_overlap: int


@dataclass
class ZoteroConfig:
    library_id: str
    library_type: str
    api_key: str


@dataclass
class SearchConfig:
    # Maximum number of search refinement rounds paper_finder can run
    # before returning its best results. 1 = no follow-up searches.
    max_search_iterations: int


@dataclass
class Config:
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig
    ingestion: IngestionConfig
    zotero: ZoteroConfig
    search: SearchConfig


def load_config(path: str = "config.yaml") -> Config:
    load_dotenv(find_dotenv(usecwd=True))

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found. Copy config.yaml.example to config.yaml and fill in your settings."
        )

    raw = yaml.safe_load(config_path.read_text())

    # Zotero — env vars take precedence over config.yaml
    zotero_raw = raw.get("zotero", {})
    library_id = os.getenv("ZOTERO_LIBRARY_ID") or zotero_raw.get("library_id", "")
    api_key = os.getenv("ZOTERO_API_KEY", "")

    if not library_id:
        raise ValueError(
            "Zotero library ID is not set. Add 'zotero.library_id' to config.yaml "
            "or set the ZOTERO_LIBRARY_ID environment variable."
        )
    if not api_key:
        raise ValueError(
            "ZOTERO_API_KEY is not set. Add it to your .env file or environment variables."
        )

    vs_raw = raw.get("vector_store", {})
    ing_raw = raw.get("ingestion", {})
    llm_raw = raw.get("llm", {})
    emb_raw = raw.get("embeddings", {})
    search_raw = raw.get("search", {})

    return Config(
        llm=LLMConfig(
            provider=llm_raw.get("provider", "openai"),
            model=llm_raw.get("model", "gpt-4o"),
            temperature=llm_raw.get("temperature", 0.2),
        ),
        embeddings=EmbeddingsConfig(
            provider=emb_raw.get("provider", "openai"),
            model=emb_raw.get("model", "text-embedding-3-small"),
        ),
        vector_store=VectorStoreConfig(
            provider=vs_raw.get("provider", "chroma"),
            persist_directory=vs_raw.get("persist_directory", ".zori/chroma"),
            qdrant_url=vs_raw.get("qdrant_url"),
        ),
        ingestion=IngestionConfig(
            mode=ing_raw.get("mode", "batch"),
            sync_on_startup=ing_raw.get("sync_on_startup", True),
            chunk_size=ing_raw.get("chunk_size", 1000),
            chunk_overlap=ing_raw.get("chunk_overlap", 200),
        ),
        zotero=ZoteroConfig(
            library_id=library_id,
            library_type=zotero_raw.get("library_type", "user"),
            api_key=api_key,
        ),
        search=SearchConfig(
            max_search_iterations=search_raw.get("max_search_iterations", 3),
        ),
    )
