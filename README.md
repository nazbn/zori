<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/nazbn/zori/master/assets/logo-dark.svg">
  <img src="https://raw.githubusercontent.com/nazbn/zori/master/assets/logo-light.svg" alt="Zori" height="48">
</picture>

An open-source multi-agent research assistant that connects to your Zotero library.
Search, summarize, and explore your research papers through a conversational interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-yellow)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![PyPI](https://img.shields.io/pypi/v/zori)

## Features

- **Hybrid search** — combines semantic vector search and BM25 keyword search over your entire library
- **Summarization** — generates structured summaries and saves them as notes directly in Zotero
- **Web UI** — clean chat interface built on FastAPI; no browser issues, works on any platform
- **Conversational context** — references like "the first one" or "that paper" are resolved across turns
- **Flexible LLM support** — OpenAI, Anthropic, or Ollama (free, runs locally)

## Web UI

```bash
zori ui
```

Open `http://localhost:7860` in your browser. The web UI is the recommended interface on Windows, where terminal hyperlinks may not render correctly.

![Zori search results](https://raw.githubusercontent.com/nazbn/zori/master/assets/search-1.png)

![Zori structured paper summary](https://raw.githubusercontent.com/nazbn/zori/master/assets/summary-cdem.png)

![Zori paper summary saved as a Zotero note](https://raw.githubusercontent.com/nazbn/zori/master/assets/summary-cdem-2.png)

## Requirements

- A Zotero account with API access
- An LLM provider: OpenAI, Anthropic, or [Ollama](https://ollama.com) (free, runs locally)

## Setup

**1. Install**

```bash
pip install zori
```

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/nazbn/zori.git
cd zori
uv sync
```

When installed from source, prefix all commands with `uv run` (e.g. `uv run zori init`, `uv run zori ingest`, `uv run zori`).
</details>

**2. Initialize**

```bash
mkdir my-zori && cd my-zori
zori init
```

`zori init` creates `config.yaml` and `.env` in the current directory. Always run `zori` from this directory.

**3. Configure**

Edit `.env` with your Zotero API key and library ID, and `config.yaml` to choose your LLM and embeddings provider (see [LLM options](#llm-options) and [Embeddings options](#embeddings-options)).

**4. Ingest your library**
```bash
zori ingest
```
Downloads your Zotero PDFs, extracts text, and builds the search index in `.zori/`.
Run time depends on library size and embedding provider. You only need to do a full ingest once.
To index new or modified items added to Zotero since the last ingest, run `zori ingest --sync`.

**5. Start the assistant**

Web UI (recommended):
```bash
zori ui
```

Or use the terminal REPL:
```bash
zori
```

## Usage

Zori supports natural language queries for searching and summarizing papers:

```
> papers on diffusion models
> papers by Vaswani
> papers from 2023 on neural radiance fields
> summarize the first one
> find attention is all you need
> summarize it
```

Queries use hybrid search (keyword + semantic). References to previous results are resolved in context (e.g. "the first one", "that paper").

In the terminal REPL: type `exit` to quit, `--new-session` to reset conversation history.

## LLM options

| Provider | `config.yaml` | Requires |
|---|---|---|
| OpenAI | `provider: openai` | `OPENAI_API_KEY` in `.env` |
| Anthropic | `provider: anthropic` | `ANTHROPIC_API_KEY` in `.env` |
| Ollama (free, local) | `provider: ollama` | [Ollama](https://ollama.com) running locally |

## Embeddings options

LLM and embeddings are configured independently — any combination works.

| Provider | `config.yaml` | Setup |
|---|---|---|
| OpenAI | `provider: openai`, `model: text-embedding-3-small` | `OPENAI_API_KEY` in `.env` |
| Ollama (free, local) | `provider: ollama`, `model: nomic-embed-text` | Ollama running + `ollama pull nomic-embed-text` |
| HuggingFace (free, local) | `provider: huggingface`, `model: <model>` (e.g. `all-MiniLM-L6-v2`) | `pip install "zori[huggingface]"` |

## Tracing (optional)

Zori supports [LangSmith](https://smith.langchain.com) tracing. To enable it, uncomment the LangSmith lines in `.env` and add your API key. Traces are sent to your LangSmith account.

## License

MIT — see [LICENSE](LICENSE).

## Contact

For questions, bug reports, or feature requests, open an issue on the [GitHub issue tracker](https://github.com/nazbn/zori/issues)
or reach out at nazanin.bagherinejad@rwth-aachen.de.

---

*This repository was developed with the assistance of [Claude](https://claude.ai) (Anthropic).*
