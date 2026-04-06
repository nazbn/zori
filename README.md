# Zori

An open-source multi-agent research assistant that connects to your Zotero library.
Search, summarize, and explore your research papers through a conversational interface.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip
- A Zotero account with API access
- OpenAI, Anthropic, or [Ollama](https://ollama.com) (free, local)

## Setup

### Install from PyPI
```bash
pip install zori
mkdir my-zori && cd my-zori
zori init
```

### Install from source
```bash
git clone https://github.com/nazbn/zori.git
cd zori
uv sync
uv run zori init
```

`zori init` creates `config.yaml` and `.env` in the current directory.
Always run `zori` from this directory.

**Configure `.env`:**
```
ZOTERO_API_KEY=...       # from zotero.org/settings/keys
ZOTERO_LIBRARY_ID=...    # your numeric user ID, shown on the same page
OPENAI_API_KEY=...       # or ANTHROPIC_API_KEY, depending on your config
```

**Configure `config.yaml`:** set your preferred LLM and embeddings provider (see [LLM options](#llm-options)).

**Ingest your library:**
```bash
zori ingest
```
Downloads your Zotero PDFs, extracts text, and builds the search index in `.zori/`.
Run time depends on library size. You only need to do a full ingest once.

**Start the assistant:**
```bash
zori
```

## Usage

Once the REPL is running, just type naturally:

```
> papers on diffusion models
> show me work by Rombach
> papers from 2023 on neural radiance fields
> summarize the first one
> find the attention paper
> summarize it
```

Zori uses hybrid search (keyword + semantic) and understands follow-up references like
"the first one" or "that last result".

Type `exit` to quit, `--new-session` to reset conversation history.

## Ingestion commands

```bash
uv run zori ingest           # full ingest (first run or rebuild)
uv run zori ingest --sync    # sync new/modified items only
```

## LLM options

| Provider | `config.yaml` | Requires |
|---|---|---|
| OpenAI | `provider: openai` | `OPENAI_API_KEY` in `.env` |
| Anthropic | `provider: anthropic` | `ANTHROPIC_API_KEY` in `.env` |
| Ollama (free, local) | `provider: ollama` | [Ollama](https://ollama.com) running locally |

The same providers are available for embeddings. If you use Ollama for both LLM and embeddings,
no API keys are needed.

## Tips

- **Debug mode:** `uv run zori --debug` prints agent decisions and search plans to help troubleshoot unexpected results.
- **HuggingFace embeddings:** install with `uv sync --extra huggingface` and set `provider: huggingface` in `config.yaml`.

## Running tests

```bash
uv sync --extra dev
uv run pytest tests/
```
