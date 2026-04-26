# Changelog

## v0.3.0 — Web UI & Ingestion Improvements

### What's new

- Web UI (`zori ui`) — clean chat interface built on FastAPI; recommended interface on all platforms
- Ingest panel with live paper progress in the UI
- Ingestion now resumes after interruption — state saved after each paper
- Embeddings batched to avoid OpenAI rate limits
  
### Removed

- Gradio dependency removed

---

## v0.2.0 — LangChain Retrieval Refactor

### What's new

- Retrieval layer rebuilt with LangChain BaseRetriever subclasses
- EnsembleRetriever replaces custom weighted RRF
- ZoriVectorStore now wraps langchain-chroma
- Author and year are now soft scoring signals instead of hard filters
- get_embeddings() returns a LangChain Embeddings object

---

## v0.1.0 — Initial Release

Initial release of Zori — a multi-agent research assistant for Zotero libraries.

### Features

- Hybrid search (BM25 + semantic) over your Zotero library
- Paper summarization with LLM
- Save summaries as notes back to Zotero
- Configurable LLM and embeddings providers (OpenAI, Anthropic, Ollama, HuggingFace)
- `zori init` for easy setup
