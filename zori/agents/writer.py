import structlog
from typing import Callable

from langchain_core.messages import AIMessage

from zori.agents.graph import ZoriState
from zori.ingestion.zotero import ZoteroClient

logger = structlog.get_logger()


def _format_note_html(summary: dict) -> str:
    """Render a summary dict as an HTML string suitable for a Zotero note."""
    contributions_html = "".join(f"<li>{c}</li>" for c in summary.get("contributions", []))
    return (
        "<h2>Zori Summary</h2>"
        f"<h3>Overview</h3><p>{summary.get('overview', '')}</p>"
        f"<h3>Key Contributions</h3><ul>{contributions_html}</ul>"
        f"<h3>Methods</h3><p>{summary.get('methods', '')}</p>"
        f"<h3>Findings</h3><p>{summary.get('findings', '')}</p>"
    )


def make_writer_node(zotero_client: ZoteroClient) -> Callable[[ZoriState], dict]:
    """Return a LangGraph node that saves the current summary as a Zotero note."""
    def writer_node(state: ZoriState) -> dict:
        logger.debug("writer_entry",
            query=state.get("query"), target_key=state.get("target_key"),
            has_summary=state.get("summary") is not None,
        )
        reply = state["query"].strip().lower()

        # User declined
        if reply not in ("yes", "y"):
            response = "No problem. What else can I help you with?"
            return {
                "pending_confirmation": False,
                "confirmation_type": None,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        # User confirmed — save summary to Zotero
        item_key = state.get("target_key")
        summary = state.get("summary")

        if not item_key or not summary:
            response = "Nothing to save — no summary in progress."
            return {
                "pending_confirmation": False,
                "confirmation_type": None,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        try:
            zotero_client.write_note(item_key, _format_note_html(summary), tags=["zori-summary"])
        except Exception as e:
            response = f"Failed to save to Zotero: {e}"
            return {
                "pending_confirmation": False,
                "confirmation_type": None,
                "response": response,
                "messages": [AIMessage(content=response)],
            }

        response = "Summary saved as a note in Zotero."
        return {
            "pending_confirmation": False,
            "confirmation_type": None,
            "summary": None,
            "response": response,
            "messages": [AIMessage(content=response)],
        }

    return writer_node
