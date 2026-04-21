from zori.retrieval.search import SearchResult


def format_authors(authors: list[str]) -> str:
    """Return a display string for an author list: full names if ≤2, 'First et al.' otherwise."""
    if not authors:
        return "Unknown"
    if len(authors) <= 2:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def zotero_link(item_key: str, label: str = "open in Zotero") -> str:
    """Return a Rich-markup hyperlink that opens the item in the Zotero desktop app."""
    uri = f"zotero://select/library/items/{item_key}"
    return f"[link={uri}]{label}[/link]"


def format_results(query: str, results: list[SearchResult]) -> str:
    """Format a list of SearchResults as a Rich display string for the REPL."""
    if not results:
        return f'Nothing in your library matched "{query}".'

    lines = [f'Found {len(results)} paper(s) matching "{query}":\n']
    for i, r in enumerate(results, 1):
        authors = format_authors(r.authors)
        year = r.year or "?"
        lines.append(f"{i}. {r.title} — {authors} ({year})")
        lines.append(f"   {zotero_link(r.item_key)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def render_response(state: dict) -> str:
    """Format a ZoriState into a display string for the CLI."""
    results = state.get("search_results", [])
    intent = state.get("intent", "")
    pending = state.get("pending_confirmation", False)

    # Search results
    if results and intent == "search" and not pending:
        return format_results(state.get("query", ""), results)

    response = state.get("response") or ""

    # Summarization — add link to the paper being summarized
    if pending and state.get("confirmation_type") == "save_summary":
        if key := state.get("target_key"):
            response = f"{response}\n{zotero_link(key)}"

    # Single paper confirmation — add link to the candidate
    elif pending and state.get("confirmation_type") == "paper_selection" and state.get("candidate_key"):
        response = f"{response}\n{zotero_link(state['candidate_key'])}"

    return response
