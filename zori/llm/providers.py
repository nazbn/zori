import os

import structlog
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

logger = structlog.get_logger()

# Providers Zori is tested with. Any other LangChain provider is passed through with a warning.
SUPPORTED_LLM_PROVIDERS = ("openai", "anthropic", "ollama")
SUPPORTED_EMBEDDING_PROVIDERS = ("openai", "huggingface", "ollama")

# Checked up front so a missing key fails at startup with one message for every provider
_API_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def get_llm(config) -> BaseChatModel:
    """Return a LangChain chat model based on config."""
    provider = config.llm.provider
    _check_provider(provider, SUPPORTED_LLM_PROVIDERS, "LLM")
    return init_chat_model(
        config.llm.model, model_provider=provider, temperature=config.llm.temperature
    )


def get_embeddings(config) -> Embeddings:
    """Return a LangChain Embeddings object based on config."""
    provider = config.embeddings.provider
    _check_provider(provider, SUPPORTED_EMBEDDING_PROVIDERS, "Embeddings")
    return init_embeddings(config.embeddings.model, provider=provider)


def _check_provider(provider: str, supported: tuple[str, ...], kind: str) -> None:
    if key := _API_KEYS.get(provider):
        _require_env(key)
    if provider not in supported:
        logger.warning(
            f"{kind} provider '{provider}' is not officially supported. "
            f"Supported providers: {', '.join(supported)}. "
            "Other providers are untested and may lack features Zori requires."
        )


def _require_env(key: str) -> None:
    if not os.getenv(key):
        raise EnvironmentError(
            f"{key} is not set. Add it to your .env file or environment variables."
        )
