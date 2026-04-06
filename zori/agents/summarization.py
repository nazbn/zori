import logging
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from zori.agents.graph import ZoriState
from zori.retrieval.formatting import format_authors, zotero_link
from zori.ingestion.zotero import ZoteroClient
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.metadata import MetadataStore

SYSTEM_PROMPT = """You are a research assistant summarizing an academic paper.
Produce a clear, structured summary for a researcher who wants to quickly understand
the paper's value and key ideas. Be concise but thorough."""


class SummaryOutput(BaseModel):
    overview: str
    contributions: list[str]
    methods: str
    findings: str


def _format_summary(title: str, authors: list[str], year: str | None,
                    item_key: str, output: SummaryOutput) -> str:
    author_str = format_authors(authors)
    link = zotero_link(item_key)
    bullets = "\n".join(f"• {c}" for c in output.contributions)

    return (
        f"{title} — {author_str} ({year or '?'})\n"
        f"{link}\n\n"
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
    structured_llm = llm.with_structured_output(SummaryOutput)

    _metadata_store: MetadataStore | None = metadata_store
    _lexical_index: LexicalIndex | None = lexical_index

    def summarization_node(state: ZoriState) -> dict:
        logger.debug("[summarization] target_key=%r", state.get("target_key"))
        nonlocal _metadata_store, _lexical_index
        if _metadata_store is None:
            _metadata_store = MetadataStore()
        if _lexical_index is None:
            _lexical_index = LexicalIndex()

        item_key = state["target_key"]
        meta = _metadata_store.get(item_key)

        if not meta:
            response = "I couldn't find metadata for that paper. It may not have been ingested yet."
            return {"response": response, "messages": [AIMessage(content=response)]}

        text = _lexical_index.get_full_text(item_key)
        if not text.strip():
            response = f'No text found for "{meta.get("title", item_key)}". Try re-ingesting the library.'
            return {"response": response, "messages": [AIMessage(content=response)]}

        try:
            output: SummaryOutput = structured_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
                SystemMessage(content=f"Paper text:\n\n{text}"),
            ])
        except Exception as e:
            response = f"Summarization failed: {e}"
            return {"response": response, "messages": [AIMessage(content=response)]}

        response = _format_summary(
            title=meta.get("title", "Unknown"),
            authors=meta.get("authors", []),
            year=meta.get("year"),
            item_key=item_key,
            output=output,
        )

        return {
            "summary": output.model_dump(),
            "response": response,
            "pending_confirmation": True,
            "confirmation_type": "save_summary",
            "messages": [AIMessage(content=response)],
        }

    return summarization_node
