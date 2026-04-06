import os
import textwrap
from pathlib import Path

import pytest

from zori.config import load_config


VALID_YAML = textwrap.dedent("""
    llm:
      provider: openai
      model: gpt-4o
      temperature: 0.2
    embeddings:
      provider: openai
      model: text-embedding-3-small
    vector_store:
      provider: chroma
      persist_directory: .zori/chroma
    ingestion:
      sync_on_startup: false
      chunk_size: 1000
      chunk_overlap: 200
    zotero:
      library_id: "123456"
      library_type: user
""")


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZOTERO_API_KEY", "test-api-key")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(VALID_YAML)
    return cfg


def test_load_valid_config(config_file):
    config = load_config()
    assert config.llm.provider == "openai"
    assert config.llm.model == "gpt-4o"
    assert config.llm.temperature == 0.2
    assert config.embeddings.model == "text-embedding-3-small"
    assert config.vector_store.provider == "chroma"
    assert config.ingestion.chunk_size == 1000
    assert config.zotero.library_id == "123456"
    assert config.zotero.api_key == "test-api-key"


def test_env_library_id_overrides_yaml(config_file, monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "999999")
    config = load_config()
    assert config.zotero.library_id == "999999"


def test_missing_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        load_config()


def test_missing_api_key(config_file, monkeypatch):
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ZOTERO_API_KEY"):
        load_config()


def test_missing_library_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZOTERO_API_KEY", "test-key")
    monkeypatch.delenv("ZOTERO_LIBRARY_ID", raising=False)
    yaml_no_id = VALID_YAML.replace('library_id: "123456"', "")
    (tmp_path / "config.yaml").write_text(yaml_no_id)
    with pytest.raises(ValueError, match="library ID"):
        load_config()
