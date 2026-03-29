# Zori

An open-source multi-agent research assistant that connects to your Zotero library.
Search, summarize, and explore your research papers through a conversational interface.

## Quickstart

_Coming soon._

## Requirements

- Python 3.10+
- A Zotero account with API access
- OpenAI or Anthropic API key

## Setup

```bash
# Install
pip install zori

# Configure
cp config.yaml.example config.yaml
cp .env.example .env
# Edit config.yaml with your Zotero library ID
# Edit .env with your API keys

# Ingest your library
zori ingest

# Start the REPL
zori
```
