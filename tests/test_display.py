import pytest

from zori.agents.paper_finder import _group_by_paper
from zori.display.rich import (
    format_authors,
    format_results,
    zotero_link,
)
from zori.retrieval.search import SearchResult


def _make_result(item_key="P1", title="Paper", score=0.9,
                 authors=None, year="2023", text="Some text."):
    return SearchResult(
        item_key=item_key,
        title=title,
        score=score,
        authors=authors or ["Alice"],
        year=year,
        journal="Journal",
        text=text,
    )


# --- format_authors ---

def test_format_authors_single():
    assert format_authors(["Alice"]) == "Alice"


def test_format_authors_two():
    assert format_authors(["Alice", "Bob"]) == "Alice, Bob"


def test_format_authors_many():
    assert format_authors(["Alice", "Bob", "Carol"]) == "Alice et al."


def test_format_authors_empty():
    assert format_authors([]) == "Unknown"


# --- zotero_link ---

def test_zotero_link_contains_uri():
    link = zotero_link("ABC123")
    assert "zotero://select/library/items/ABC123" in link


def test_zotero_link_contains_label():
    link = zotero_link("ABC123", label="open")
    assert "open" in link


# --- _group_by_paper ---

def test_group_by_paper_deduplicates():
    results = [
        _make_result(item_key="P1", score=0.9),
        _make_result(item_key="P1", score=0.7),
        _make_result(item_key="P2", score=0.8),
    ]
    grouped = _group_by_paper(results)
    keys = [r.item_key for r in grouped]
    assert keys.count("P1") == 1
    assert keys.count("P2") == 1


def test_group_by_paper_keeps_highest_score():
    results = [
        _make_result(item_key="P1", score=0.7),
        _make_result(item_key="P1", score=0.9),
    ]
    grouped = _group_by_paper(results)
    assert grouped[0].score == 0.9


def test_group_by_paper_sorted_by_score():
    results = [
        _make_result(item_key="P1", score=0.5),
        _make_result(item_key="P2", score=0.9),
        _make_result(item_key="P3", score=0.7),
    ]
    grouped = _group_by_paper(results)
    scores = [r.score for r in grouped]
    assert scores == sorted(scores, reverse=True)


# --- format_results ---

def test_format_results_no_results():
    output = format_results("transformers", [])
    assert "Nothing" in output
    assert "transformers" in output


def test_format_results_shows_title():
    results = [_make_result(title="Attention is All You Need")]
    output = format_results("transformers", results)
    assert "Attention is All You Need" in output


def test_format_results_shows_year():
    results = [_make_result(year="2017")]
    output = format_results("transformers", results)
    assert "2017" in output


def test_format_results_contains_zotero_link():
    results = [_make_result(item_key="XYZ789")]
    output = format_results("transformers", results)
    assert "XYZ789" in output
