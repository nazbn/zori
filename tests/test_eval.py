"""DeepEval retrieval quality tests.

Run with:
    deepeval test run tests/test_eval.py
or:
    uv run pytest tests/test_eval.py -v

Requires:
    uv sync --extra eval
    scripts/eval_dataset.yaml (copy from scripts/eval_dataset.yaml.example and fill in)
"""
import pytest
import yaml
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

try:
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
except ImportError:
    pytest.skip("deepeval not installed — run: uv sync --extra eval", allow_module_level=True)

DATASET_PATH = Path("scripts/eval_dataset.yaml")
if not DATASET_PATH.exists():
    pytest.skip(
        f"{DATASET_PATH} not found — copy from scripts/eval_dataset.yaml.example and fill in",
        allow_module_level=True,
    )

if not Path(".zori/state.json").exists():
    pytest.skip("Library not ingested — run: zori ingest", allow_module_level=True)

from zori.config import load_config
from zori.llm.providers import get_embeddings
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.metadata import MetadataStore
from zori.retrieval.search import SearchService
from zori.retrieval.vector import create_vector_store

# ---------------------------------------------------------------------------
# Services — initialized once for all test cases
# ---------------------------------------------------------------------------

_config = load_config()
_metadata_store = MetadataStore()
_lexical_index = LexicalIndex()
_vector_store = create_vector_store(
    _config.vector_store.persist_directory, get_embeddings(_config)
)
_search_service = SearchService(_vector_store, _metadata_store, _lexical_index)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(item_key: str) -> str:
    """Build context string for an item: title, year, authors, abstract."""
    meta = _metadata_store.get(item_key) or {}
    title = meta.get("title", "Unknown")
    year = meta.get("year", "")
    authors = meta.get("authors", [])
    abstract = _lexical_index.get_abstract(item_key)
    header = f"{title} ({year})" if year else title
    author_line = f"Authors: {', '.join(authors)}" if authors else ""
    return "\n".join(p for p in [header, author_line, abstract] if p)


def _make_test_case(entry: dict) -> LLMTestCase:
    query = entry["query"]
    results = _search_service.hybrid_search(
        lexical_queries=[query],
        semantic_query=query,
        author=entry.get("author"),
        year_from=str(entry["year_from"]) if entry.get("year_from") else None,
        year_to=str(entry["year_to"]) if entry.get("year_to") else None,
        tags=entry.get("tags"),
    )
    contexts = [_build_context(r.item_key) for r in results]
    actual_output = "\n".join(f"- {r.title} ({r.year or '?'})" for r in results)
    return LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=entry["expected_output"],
        retrieval_context=contexts,
    )


# ---------------------------------------------------------------------------
# Test cases — built from dataset at collection time
# ---------------------------------------------------------------------------

with open(DATASET_PATH) as f:
    _dataset = yaml.safe_load(f)

_test_cases = [_make_test_case(entry) for entry in _dataset]

_metrics = [
    ContextualRelevancyMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
]


@pytest.mark.parametrize("test_case", _test_cases)
def test_retrieval(test_case: LLMTestCase):
    assert_test(test_case, _metrics)
