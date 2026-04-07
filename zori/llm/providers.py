import os
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(config) -> BaseChatModel:
    """Return a LangChain chat model based on config."""
    provider = config.llm.provider
    model = config.llm.model
    temperature = config.llm.temperature

    if provider == "openai":
        _require_env("OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        _require_env("ANTHROPIC_API_KEY")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown LLM provider '{provider}'. Choose from: openai, anthropic, ollama"
    )


def get_embed_fn(config) -> Callable[[list[str]], list[list[float]]]:
    """Return an embedding function compatible with ChromaVectorStore."""
    provider = config.embeddings.provider
    model = config.embeddings.model

    if provider == "openai":
        _require_env("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=model)
        return embeddings.embed_documents

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=model)
        return embeddings.embed_documents

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=model)
        return embeddings.embed_documents

    raise ValueError(
        f"Unknown embeddings provider '{provider}'. Choose from: openai, huggingface, ollama"
    )


def _require_env(key: str) -> None:
    if not os.getenv(key):
        raise EnvironmentError(
            f"{key} is not set. Add it to your .env file or environment variables."
        )
