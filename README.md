# Zori

An open-source multi-agent research assistant that connects to your Zotero library.
Search, summarize, and explore your research papers through a conversational interface.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip
- A Zotero account with API access
- OpenAI, Anthropic, or [Ollama](https://ollama.com) (free, local)

## Setup

**1. Clone and install**
```bash
git clone https://github.com/nazbn/zori.git
cd zori
uv sync
```

**2. Configure**
```bash
cp config.yaml.example config.yaml
cp .env.example .env
```

Edit `config.yaml` with your Zotero library ID and preferred LLM.
Edit `.env` with your API keys.

**3. Ingest your library**
```bash
uv run zori ingest
```

**4. Start the assistant**
```bash
uv run zori
```

## One-shot commands

```bash
uv run zori ingest           # ingest full library
uv run zori ingest --sync    # sync new items only
uv run zori search "query"   # quick search without entering the REPL
```

## LLM options

| Provider | Config | Requires |
|---|---|---|
| OpenAI | `provider: openai` | `OPENAI_API_KEY` in `.env` |
| Anthropic | `provider: anthropic` | `ANTHROPIC_API_KEY` in `.env` |
| Ollama (free, local) | `provider: ollama` | [Ollama](https://ollama.com) running locally |

## Running tests

```bash
uv sync --extra dev
uv run pytest tests/
```
