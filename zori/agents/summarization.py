import structlog
from typing import Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel

logger = structlog.get_logger()

from zori.agents.confirm import is_no, is_yes, redirect_to_router
from zori.agents.graph import ZoriState
from zori.display.rich import format_authors
from zori.ingestion.zotero import ZoteroClient
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.metadata import MetadataStore


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

SYSTEM_PROMPT = """You are a research assistant summarizing an academic paper.
Produce a clear, structured summary for a researcher who wants to quickly understand
the paper's value and key ideas. Be concise but thorough."""


class SummaryOutput(BaseModel):
    overview: str
    contributions: list[str]
    methods: str
    findings: str


def _format_summary(title: str, authors: list[str], year: str | None,
                    output: SummaryOutput) -> str:
    """Render a SummaryOutput as plain text with a save prompt."""
    author_str = format_authors(authors)
    bullets = "\n".join(f"• {c}" for c in output.contributions)

    return (
        f"{title} — {author_str} ({year or '?'})\n\n"
        f"Overview\n{output.overview}\n\n"
        f"Key Contributions\n{bullets}\n\n"
        f"Methods\n{output.methods}\n\n"
        f"Findings\n{output.findings}\n\n"
        f"Would you like to save this summary to Zotero? (yes / no)"
    )


def make_summarization_node(
    zotero_client: ZoteroClient,
    llm: BaseChatModel,
    metadata_store: MetadataStore | None = None,
    lexical_index: LexicalIndex | None = None,
) -> Callable[[ZoriState], dict]:
    """Return a LangGraph node that reads full paper text and produces a structured summary.

    Stores the summary in state and stops — the save confirmation lives in confirm_save,
    so resuming after the interrupt doesn't re-run the LLM.
    """
    structured_llm = llm.with_structured_output(SummaryOutput)

    _metadata_store: MetadataStore | None = metadata_store
    _lexical_index: LexicalIndex | None = lexical_index

    def summarization_node(state: ZoriState) -> dict:
        logger.debug("summarization_entry", target_key=state.get("target_key"))
        nonlocal _metadata_store, _lexical_index
        if _metadata_store is None:
            _metadata_store = MetadataStore()
        if _lexical_index is None:
            _lexical_index = LexicalIndex()

        item_key = state["target_key"]
        meta = _metadata_store.get(item_key)

        if not meta:
            response = "I couldn't find metadata for that paper. It may not have been ingested yet."
            return {"summary": None, "response": response, "messages": [AIMessage(content=response)]}

        text = _lexical_index.get_full_text(item_key)
        if not text.strip():
            response = f'No text found for "{meta.get("title", item_key)}". Try re-ingesting the library.'
            return {"summary": None, "response": response, "messages": [AIMessage(content=response)]}

        try:
            output: SummaryOutput = structured_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
                SystemMessage(content=f"Paper text:\n\n{text}"),
            ])
        except Exception as e:
            response = f"Summarization failed: {e}"
            return {"summary": None, "response": response, "messages": [AIMessage(content=response)]}

        summary_text = _format_summary(
            title=meta.get("title", "Unknown"),
            authors=meta.get("authors", []),
            year=meta.get("year"),
            output=output,
        )
        return {"summary": output.model_dump(), "summary_text": summary_text}

    return summarization_node


def make_confirm_save_node(zotero_client: ZoteroClient) -> Callable[[ZoriState], dict]:
    """Return a LangGraph node that asks whether to save the summary, then writes it to Zotero.

    Runs after summarization. Everything it needs is already in state, so when LangGraph
    re-runs this node on resume, nothing expensive happens twice.
    """

    def confirm_save_node(state: ZoriState) -> Command[Literal["router", "__end__"]]:
        item_key = state["target_key"]
        question = state["summary_text"]
        reply = interrupt({"text": question, "zotero_key": item_key})

        if is_yes(reply):
            try:
                zotero_client.write_note(item_key, _format_note_html(state["summary"]), tags=["zori-summary"])
                response = "Summary saved as a note in Zotero."
            except Exception as e:
                response = f"Failed to save to Zotero: {e}"
        elif is_no(reply):
            response = "No problem. What else can I help you with?"
        else:
            # e.g. "no, summarize the other one instead" — handle it as a new request
            return redirect_to_router(question, reply)

        return Command(goto=END, update={"response": response, "messages": [AIMessage(content=response)]})

    return confirm_save_node
