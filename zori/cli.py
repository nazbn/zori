import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")

from dataclasses import dataclass

from pathlib import Path

import typer
from langchain_core.messages import HumanMessage
from rich.console import Console

from zori.agents.graph import ZoriState, build_graph
from zori.config import load_config
from zori.log import configure_logging
from zori.display.rich import render_response
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
    pipeline = IngestionPipeline(zotero, vector_store, metadata_store, lexical_index)

    return Services(zotero, metadata_store, search_service, pipeline, lexical_index), config


def _fresh_state() -> dict:
    return {
        "messages": [],
        "query": "",
        "intent": "",
        "target_key": None,
        "display_query": "",
        "search_results": [],
        "summary": None,
        "response": None,
        "pending_confirmation": False,
        "confirmation_type": None,
        "candidate_key": None,
    }


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
):
    """Launch the interactive REPL. Type your query to search or summarize papers.
    Use 'exit' to quit, '--new-session' to reset conversation history."""
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    configure_logging(debug=debug)
    if ctx.invoked_subcommand is None:
        _repl()


def _repl():
    try:
        services, config = _init_services()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Setup error:[/red] {e}")
        raise typer.Exit(1)

    llm = get_llm(config)
    graph = build_graph(services.search_service, services.zotero, llm, services.lexical_index)

    if config.ingestion.sync_on_startup:
        console.print("[dim]Checking for new items in Zotero...[/dim]")
        services.pipeline.run_sync()

    if not Path(".zori/state.json").exists():
        console.print("[yellow]Library not ingested yet. Run [bold]zori ingest[/bold] to get started.[/yellow]\n")

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

        if output := render_response(state):
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
def ingest(sync: bool = typer.Option(False, "--sync", help="Sync new/modified items only (skips items already ingested).")):
    """Ingest your Zotero library. Downloads PDFs, extracts text, and builds the search index.
    Run without --sync for a full ingest, or with --sync to pick up new items since last run."""
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
def ui(debug: bool = typer.Option(False, "--debug", help="Enable debug logging.")):
    """Launch the Zori web UI in your browser."""
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    from zori.ui.server import launch
    launch(debug=debug)

