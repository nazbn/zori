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
    """Format a ZoriState into a markdown string for the Gradio UI."""
    results = state.get("search_results", [])
    intent = state.get("intent", "")
    pending = state.get("pending_confirmation", False)

    # Search results
    if results and intent == "search" and not pending:
        return format_results_md(state.get("display_query") or state.get("query", ""), results)

    response = state.get("response") or ""

    # Summarization — add link to the paper being summarized
    if pending and state.get("confirmation_type") == "save_summary":
        if key := state.get("target_key"):
            response = f"{response}\n\n{zotero_link_md(key)}"

    # Single paper confirmation — add link to the candidate
    elif pending and state.get("confirmation_type") == "paper_selection" and state.get("candidate_key"):
        response = f"{response}\n\n{zotero_link_md(state['candidate_key'])}"

    return response
