from typing import Callable

from langchain_core.messages import AIMessage

from zori.agents.graph import ZoriState
from zori.agents.retrieval import format_results, search_library, zotero_link
from zori.retrieval.search import SearchService

CONFIRMATION_THRESHOLD = 0.75  # minimum score to auto-propose a paper for summarize


def make_paper_finder_node(search_service: SearchService) -> Callable[[ZoriState], dict]:
    def paper_finder_node(state: ZoriState) -> dict:
        # --- Handle pending confirmation (user replied yes/no) ---
        if state.get("pending_confirmation"):
            return _handle_confirmation(state)

        query = state["query"]
        mode = state.get("search_mode", "display")
        results = search_library(search_service, query)

        # --- Display mode (intent="search") ---
        if mode == "display":
            response = format_results(query, results)
            return {
                "search_results": results,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        # --- Find for summarize mode (intent="summarize", no key yet) ---
        if not results:
            response = f'I couldn\'t find "{query}" in your library.'
            return {
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        top = results[0]

        if top.score >= CONFIRMATION_THRESHOLD:
            # Confident enough to propose without asking
            authors = top.authors[0] if top.authors else "Unknown"
            year = top.year or "?"
            link = zotero_link(top.item_key)
            response = (
                f'I found "{top.title}" by {authors} ({year}). {link}\n'
                f"Shall I summarize this one? (yes / no)"
            )
            return {
                "search_results": results,
                "pending_confirmation": True,
                "candidate_key": top.item_key,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        # Multiple plausible results — ask user to pick
        lines = [f'I found several papers that could match "{query}". Which one?\n']
        for i, r in enumerate(results[:3], 1):
            authors = r.authors[0] if r.authors else "Unknown"
            lines.append(f"  {i}. {r.title} — {authors} ({r.year or '?'}) {zotero_link(r.item_key)}")
        lines.append("\nReply with a number, or 'none' to cancel.")
        response = "\n".join(lines)

        return {
            "search_results": results,
            "pending_confirmation": True,
            "candidate_key": None,   # user must pick
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    return paper_finder_node


def _handle_confirmation(state: ZoriState) -> dict:
    """Process the user's yes/no or numbered reply."""
    reply = state["query"].strip().lower()
    results = state.get("search_results", [])

    # Numbered pick
    if reply.isdigit():
        idx = int(reply) - 1
        if 0 <= idx < len(results):
            key = results[idx].item_key
            return {
                "target_key": key,
                "pending_confirmation": False,
                "candidate_key": None,
            }

    # Yes/no confirmation
    if reply in ("yes", "y"):
        return {
            "target_key": state.get("candidate_key"),
            "pending_confirmation": False,
            "candidate_key": None,
        }

    # Cancelled or unrecognised
    response = "No problem. What else can I help you with?"
    return {
        "target_key": None,
        "pending_confirmation": False,
        "candidate_key": None,
        "response": response,
        "messages": [AIMessage(content=response)],
    }
