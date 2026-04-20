# Zori

An open-source multi-agent research assistant that connects to your Zotero library.
Search, summarize, and explore your research papers through a conversational interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-yellow)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![PyPI](https://img.shields.io/pypi/v/zori)

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

Type `exit` to quit, `--new-session` to reset conversation history.

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

## License

MIT — see [LICENSE](LICENSE).

## Contact

For questions, bug reports, or feature requests, open an issue on the [GitHub issue tracker](https://github.com/nazbn/zori/issues)
or reach out at nazanin.bagherinejad@rwth-aachen.de.

---

*This repository was developed with the assistance of [Claude](https://claude.ai) (Anthropic).*

