from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from zori.llm.providers import get_embeddings, get_llm


def _config(llm_provider="openai", llm_model="gpt-4o", emb_provider="openai", emb_model="text-embedding-3-small"):
    return SimpleNamespace(
        llm=SimpleNamespace(provider=llm_provider, model=llm_model, temperature=0.2),
        embeddings=SimpleNamespace(provider=emb_provider, model=emb_model),
    )


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


# --- get_llm ---

@pytest.mark.parametrize("provider, model, cls", [
    ("openai", "gpt-4o", ChatOpenAI),
    ("anthropic", "claude-sonnet-4-6", ChatAnthropic),
    ("ollama", "llama3.1", ChatOllama),
])
def test_supported_llm_providers_build_their_model(keys, provider, model, cls):
    llm = get_llm(_config(llm_provider=provider, llm_model=model))
    assert isinstance(llm, cls)
    assert llm.temperature == 0.2


@pytest.mark.parametrize("provider, key", [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
def test_missing_api_key_fails_with_env_message(monkeypatch, provider, key):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(EnvironmentError, match=f"{key} is not set"):
        get_llm(_config(llm_provider=provider))


def test_unsupported_llm_provider_warns_and_passes_through():
    with patch("zori.llm.providers.init_chat_model") as mock_init, \
         patch("zori.llm.providers.logger") as mock_logger:
        get_llm(_config(llm_provider="mistralai", llm_model="mistral-large-latest"))
    mock_init.assert_called_once_with("mistral-large-latest", model_provider="mistralai", temperature=0.2)
    message = mock_logger.warning.call_args[0][0]
    assert "LLM provider 'mistralai' is not officially supported" in message
    assert "openai, anthropic, ollama" in message


def test_supported_llm_provider_does_not_warn(keys):
    with patch("zori.llm.providers.logger") as mock_logger:
        get_llm(_config(llm_provider="openai"))
    mock_logger.warning.assert_not_called()


# --- get_embeddings ---

@pytest.mark.parametrize("provider, model, cls", [
    ("openai", "text-embedding-3-small", OpenAIEmbeddings),
    ("ollama", "nomic-embed-text", OllamaEmbeddings),
])
def test_supported_embedding_providers_build_their_model(keys, provider, model, cls):
    assert isinstance(get_embeddings(_config(emb_provider=provider, emb_model=model)), cls)


def test_huggingface_embeddings_built_with_model_name():
    # Stand-in class: the real one downloads the model on construction
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_cls, \
         patch("zori.llm.providers.logger") as mock_logger:
        get_embeddings(_config(emb_provider="huggingface", emb_model="all-MiniLM-L6-v2"))
    mock_cls.assert_called_once_with(model_name="all-MiniLM-L6-v2")
    mock_logger.warning.assert_not_called()


def test_embeddings_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY is not set"):
        get_embeddings(_config(emb_provider="openai"))


def test_unsupported_embedding_provider_warns_and_passes_through():
    with patch("zori.llm.providers.init_embeddings") as mock_init, \
         patch("zori.llm.providers.logger") as mock_logger:
        get_embeddings(_config(emb_provider="cohere", emb_model="embed-english-v3.0"))
    mock_init.assert_called_once_with("embed-english-v3.0", provider="cohere")
    assert "Embeddings provider 'cohere' is not officially supported" in mock_logger.warning.call_args[0][0]
