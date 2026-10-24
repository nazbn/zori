from zori.display.rich import format_authors
from zori.retrieval.search import SearchResult


def zotero_link_md(item_key: str, label: str = "Open in Zotero") -> str:
    """Return a markdown hyperlink that opens the item in the Zotero desktop app."""
    uri = f"zotero://select/library/items/{item_key}"
    return f"[{label}]({uri})"


def format_results_md(query: str, results: list[SearchResult]) -> str:
    """Format a list of SearchResults as markdown for the Gradio UI."""
    if not results:
        return f'Nothing in your library matched "{query}".'

    lines = [f'Found {len(results)} paper(s) matching **"{query}"**:\n']
    for i, r in enumerate(results, 1):
        authors = format_authors(r.authors)
        year = r.year or "?"
        lines.append(f"{i}. **{r.title}** — {authors} ({year})  ")
        lines.append(f"   {zotero_link_md(r.item_key)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def render_response_md(state: dict) -> str:
    """Format a completed ZoriState into a markdown string for the web UI."""
    results = state.get("search_results", [])
    intent = state.get("intent", "")

    if results and intent == "search":
        return format_results_md(state.get("display_query") or state.get("query", ""), results)

    return state.get("response") or ""


def render_interrupt_md(value: dict | str) -> str:
    """Format an interrupt value (question to the user) for the web UI, adding a Zotero link if provided."""
    if isinstance(value, dict):
        text = value.get("text", "")
        if key := value.get("zotero_key"):
            return f"{text}\n\n{zotero_link_md(key)}"
        return text
    return str(value)
