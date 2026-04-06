from zori.retrieval.search import SearchResult


def format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) <= 2:
        return ", ".join(authors)
    return f"{authors[0]} et al."


def zotero_link(item_key: str, label: str = "open in Zotero") -> str:
    uri = f"zotero://select/library/items/{item_key}"
    return f"[link={uri}]{label}[/link]"


def format_results(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f'Nothing in your library matched "{query}".'

    lines = [f'Found {len(results)} paper(s) matching "{query}":\n']
    for i, r in enumerate(results, 1):
        authors = format_authors(r.authors)
        year = r.year or "?"
        lines.append(f"{i}. {r.title} — {authors} ({year})")
        # TODO: re-enable once section titles filter out reference chunks (v2)
        # if r.text:
        #     preview = r.text[:150].strip().replace("\n", " ")
        #     if len(r.text) > 150:
        #         preview += "..."
        #     lines.append(f'   "{preview}"')
        lines.append(f"   {zotero_link(r.item_key)}")
        lines.append("")

    return "\n".join(lines).rstrip()
