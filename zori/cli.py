from dataclasses import dataclass

import typer
from langchain_core.messages import HumanMessage
from rich.console import Console

from zori.agents.graph import ZoriState, build_graph
from zori.agents.retrieval import format_results
from zori.config import load_config
from zori.ingestion.pipeline import IngestionPipeline
from zori.ingestion.zotero import ZoteroClient
from zori.llm.client import get_embed_fn, get_llm
from zori.retrieval.search import SearchService
from zori.retrieval.vectorstore import MetadataStore, create_vector_store

app = typer.Typer(help="Zori — your personal research assistant.")
console = Console()


@dataclass
class Services:
    zotero: ZoteroClient
    metadata_store: MetadataStore
    search_service: SearchService
    pipeline: IngestionPipeline


def _init_services() -> tuple[Services, object]:
    config = load_config()

    zotero = ZoteroClient(
        library_id=config.zotero.library_id,
        library_type=config.zotero.library_type,
        api_key=config.zotero.api_key,
    )

    embed_fn = get_embed_fn(config)
    vector_store = create_vector_store(config.vector_store.persist_directory, embed_fn)
    metadata_store = MetadataStore()
    search_service = SearchService(vector_store, metadata_store)
    pipeline = IngestionPipeline(zotero, vector_store, metadata_store)

    return Services(zotero, metadata_store, search_service, pipeline), config


def _fresh_state() -> dict:
    return {
        "messages": [],
        "query": "",
        "intent": "",
        "search_mode": "display",
        "target_key": None,
        "search_results": [],
        "summary": None,
        "response": None,
        "pending_confirmation": False,
        "confirmation_type": None,
        "candidate_key": None,
    }


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Launch the interactive REPL."""
    if ctx.invoked_subcommand is None:
        _repl()


def _repl():
    try:
        services, config = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    llm = get_llm(config)
    graph = build_graph(services.search_service, services.zotero, llm)

    if config.ingestion.sync_on_startup:
        console.print("[dim]Checking for new items in Zotero...[/dim]")
        services.pipeline.run_sync()

    console.print("Zori ready. Type [bold]exit[/bold] to quit, [bold]--new-session[/bold] to start fresh.\n")

    state = _fresh_state()

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "exit":
            console.print("Goodbye!")
            break

        if query.lower() == "--new-session":
            state = _fresh_state()
            console.print("[dim]Session cleared. Starting fresh.[/dim]\n")
            continue

        state["query"] = query
        state["messages"] = state["messages"] + [HumanMessage(content=query)]

        try:
            state = graph.invoke(state)
        except Exception as e:
            console.print(f"[red]Something went wrong:[/red] {e}")
            continue

        if state.get("response"):
            console.print(state["response"])
        console.print()


@app.command()
def ingest(sync: bool = typer.Option(False, "--sync", help="Sync new items only.")):
    """Ingest your Zotero library into the vector store."""
    try:
        services, _ = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    if sync:
        services.pipeline.run_sync()
    else:
        services.pipeline.run_full()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    top: int = typer.Option(5, "--top", help="Number of results to return."),
):
    """Search your Zotero library."""
    try:
        services, _ = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    results = services.search_service.search(query, top_k=top)
    console.print(format_results(query, results))
