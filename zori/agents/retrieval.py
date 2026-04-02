# Internal library search helpers used by paper_finder.
# Not a graph node — paper_finder is the node that calls these.

from zori.retrieval.search import SearchResult, SearchService

TOP_K = 10  # fetch more than displayed to improve grouping quality
MAX_DISPLAY = 5


def search_library(search_service: SearchService, query: str) -> list[SearchResult]:
    """Search the vector store and return results grouped by paper."""
    raw = search_service.search(query, top_k=TOP_K)
    return _group_by_paper(raw)[:MAX_DISPLAY]


def _group_by_paper(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-scoring chunk per paper."""
    seen: dict[str, SearchResult] = {}
    for result in results:
        if result.item_key not in seen or result.score > seen[result.item_key].score:
            seen[result.item_key] = result
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)


def format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= 2:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def zotero_link(item_key: str, label: str = "open in Zotero") -> str:
    uri = f"zotero://select/library/items/{item_key}"
    return f"\x1b]8;;{uri}\x1b\\{label}\x1b]8;;\x1b\\"


def format_results(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f'Nothing in your library matched "{query}".'

    lines = [f'Found {len(results)} paper(s) matching "{query}":\n']
    for i, r in enumerate(results, 1):
        authors = format_authors(r.authors)
        year = r.year or "?"
        preview = r.text[:150].strip().replace("\n", " ")
        if len(r.text) > 150:
            preview += "..."
        lines.append(f"{i}. {r.title} — {authors} ({year})")
        lines.append(f'   "{preview}"')
        lines.append(f"   {zotero_link(r.item_key)}")
        lines.append("")

    return "\n".join(lines).rstrip()
