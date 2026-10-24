import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")

from dataclasses import dataclass

from pathlib import Path

import uuid

import typer
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console

from zori.agents.graph import ZoriState, build_graph
from zori.config import load_config
from zori.log import configure_logging
from zori.display.rich import render_interrupt, render_response
from zori.ingestion.pdf import PDFParser
from zori.ingestion.pipeline import IngestionPipeline
from zori.ingestion.zotero import ZoteroClient
from zori.llm.providers import get_embeddings, get_llm
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.search import SearchService
from zori.retrieval.metadata import MetadataStore
from zori.retrieval.vector import create_vector_store

app = typer.Typer(help="Zori — multi-agent research assistant for your Zotero library.")
console = Console()


@dataclass
class Services:
    zotero: ZoteroClient
    metadata_store: MetadataStore
    search_service: SearchService
    pipeline: IngestionPipeline
    lexical_index: LexicalIndex


def _init_services() -> tuple[Services, object]:
    config = load_config()

    zotero = ZoteroClient(
        library_id=config.zotero.library_id,
        library_type=config.zotero.library_type,
        api_key=config.zotero.api_key,
    )

    embeddings = get_embeddings(config)
    vector_store = create_vector_store(config.vector_store.persist_directory, embeddings)
    metadata_store = MetadataStore()
    lexical_index = LexicalIndex()
    search_service = SearchService(vector_store, metadata_store, lexical_index)
    parser = PDFParser(config.ingestion.chunk_size, config.ingestion.chunk_overlap)
    pipeline = IngestionPipeline(zotero, vector_store, metadata_store, lexical_index, parser=parser)

    return Services(zotero, metadata_store, search_service, pipeline, lexical_index), config



@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    """Launch the interactive REPL. Type your query to search or summarize papers.
    Use 'exit' to quit, '--new-session' to reset conversation history."""
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    if ctx.invoked_subcommand is None:
        configure_logging(debug=debug)
        _repl()


def _repl():
    try:
        services, config = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    llm = get_llm(config)
    graph = build_graph(
        services.search_service, services.zotero, llm, services.lexical_index, services.metadata_store,
    )

    if config.ingestion.sync_on_startup:
        console.print("[dim]Checking for new items in Zotero...[/dim]")
        services.pipeline.run_sync()

    if not Path(".zori/state.json").exists():
        console.print("[yellow]Library not ingested yet. Run [bold]zori ingest[/bold] to get started.[/yellow]\n")

    console.print("Zori ready. Type [bold]exit[/bold] to quit, [bold]--new-session[/bold] to start fresh.\n")

    thread_id = str(uuid.uuid4())
    config_dict = {"configurable": {"thread_id": thread_id}}

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
            thread_id = str(uuid.uuid4())
            config_dict = {"configurable": {"thread_id": thread_id}}
            console.print("[dim]Session cleared. Starting fresh.[/dim]\n")
            continue

        try:
            # Resume only if a question is pending. snapshot.next alone isn't enough: a node
            # that crashed also leaves next set, with no question for this reply to answer.
            if graph.get_state(config_dict).interrupts:
                result = graph.invoke(Command(resume=query), config=config_dict)
            else:
                result = graph.invoke(
                    {"messages": [HumanMessage(content=query)], "query": query},
                    config=config_dict,
                )
        except Exception as e:
            console.print(f"[red]Something went wrong:[/red] {e}")
            continue

        if "__interrupt__" in result:
            output = render_interrupt(result["__interrupt__"][0].value)
        else:
            output = render_response(result)

        if output:
            console.print(output, markup=True)
        console.print()


@app.command()
def init():
    """Create config.yaml and .env in the current directory from built-in templates."""
    import importlib.resources as pkg_resources
    import shutil

    templates = pkg_resources.files("zori.templates")
    created = []
    skipped = []

    for filename, dest in [("config.yaml.example", "config.yaml"), (".env.example", ".env")]:
        target = Path(dest)
        if target.exists():
            skipped.append(dest)
        else:
            src = templates.joinpath(filename)
            target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            created.append(dest)

    for f in created:
        console.print(f"[green]Created[/green] {f}")
    for f in skipped:
        console.print(f"[yellow]Skipped[/yellow] {f} (already exists)")

    if created:
        console.print("\nNext steps:")
        console.print("  1. Edit [bold].env[/bold] with your Zotero API key and library ID")
        console.print("  2. Edit [bold]config.yaml[/bold] to choose your LLM provider")
        console.print("  3. Run [bold]zori ingest[/bold] to index your library")
        console.print("  4. Run [bold]zori[/bold] to start the assistant")


@app.command()
def ingest(
    sync: bool = typer.Option(False, "--sync", help="Sync new/modified items only."),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Ingest only items in this collection (by name)."),
):
    """Ingest your Zotero library. Downloads PDFs, extracts text, and builds the search index.
    Run without --sync for a full ingest, or with --sync to pick up new items since last run.
    Use --collection to scope ingestion to a specific collection."""
    configure_logging(debug=False)
    try:
        services, _ = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    collection_key: str | None = None
    if collection:
        col = services.zotero.find_collection(collection)
        if col is None:
            console.print(f"[red]Collection not found:[/red] {collection!r}")
            console.print("Run [bold]zori collections[/bold] to see available collections.")
            raise typer.Exit(1)
        collection_key = col.key

    if sync:
        services.pipeline.run_sync(collection_key=collection_key, collection_name=collection)
    else:
        services.pipeline.run_full(collection_key=collection_key, collection_name=collection)


@app.command()
def collections():
    """List all collections in your Zotero library."""
    configure_logging(debug=False)
    try:
        services, _ = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    cols = services.zotero.fetch_collections()
    if not cols:
        console.print("No collections found.")
        return

    def _print_level(parent_key: str | None, indent: int) -> None:
        children = sorted(
            [c for c in cols if c.parent_key == parent_key],
            key=lambda c: c.name.lower(),
        )
        for col in children:
            n = col.num_items
            label = f"{'item' if n == 1 else 'items'}"
            prefix = "  " + "  " * indent + ("└─ " if indent > 0 else "")
            name = f"[bold]{col.name}[/bold]" if indent == 0 else col.name
            console.print(f"{prefix}{name}  [dim]({n} {label})[/dim]")
            _print_level(col.key, indent + 1)

    _print_level(None, 0)


@app.command()
def ui(debug: bool = typer.Option(False, "--debug", help="Enable debug logging.")):
    """Launch the Zori web UI in your browser."""
    from zori.ui.server import launch
    launch(debug=debug)

