import json
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from zori.ingestion.pdf import PDFParser
from zori.ingestion.zotero import ZoteroClient, ZoteroItem
from zori.retrieval.lexical import LexicalIndex
from zori.retrieval.metadata import MetadataStore
from zori.retrieval.vector import ChromaVectorStore

console = Console()

STATE_PATH = Path(".zori/state.json")


@dataclass
class IngestionResult:
    ingested: int = 0
    skipped_no_pdf: int = 0
    skipped_already_done: int = 0
    failed: int = 0
    failed_keys: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        zotero: ZoteroClient,
        vector_store: ChromaVectorStore,
        metadata_store: MetadataStore,
        lexical_index: LexicalIndex | None = None,
    ):
        self._zotero = zotero
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._lexical_index = lexical_index
        self._parser = PDFParser()
        self._state = self._load_state()

    def run_full(self) -> IngestionResult:
        """Ingest the entire library, re-ingesting items whose version has changed."""
        console.print("Fetching items from Zotero...")
        items = self._zotero.fetch_all_items()
        console.print(f"Ingesting {len(items)} items from Zotero...\n")
        result = self._process_items(items)
        self._save_state()
        self._print_summary(result)
        return result

    def run_sync(self) -> IngestionResult:
        """Ingest only items added or changed since the last sync."""
        since = self._state.get("library_version", 0)
        console.print(f"Checking for new or changed items since version {since}...")
        items = self._zotero.fetch_new_items(since_version=since)
        if not items:
            console.print("Nothing new.")
            return IngestionResult()
        console.print(f"Ingesting {len(items)} new or updated items...\n")
        result = self._process_items(items)
        self._save_state()
        self._print_summary(result)
        return result

    def _process_items(self, items: list[ZoteroItem]) -> IngestionResult:
        result = IngestionResult()

        for item in items:
            if not item.attachments:
                console.print(f"  [yellow]—[/yellow] {item.title[:70]} — no PDF, skipping")
                result.skipped_no_pdf += 1
                continue

            current_version = item.version
            stored_version = self._state.get("ingested", {}).get(item.key)

            if stored_version is not None and stored_version == current_version:
                result.skipped_already_done += 1
                continue

            try:
                self._ingest_item(item)
                self._state.setdefault("ingested", {})[item.key] = current_version
                console.print(f"  [green]✓[/green] {item.title[:70]}")
                result.ingested += 1
            except Exception as e:
                console.print(f"  [red]✗[/red] {item.title[:70]} — {e}")
                result.failed += 1
                result.failed_keys.append(item.key)

        library_version = self._zotero.get_library_version()
        self._state["library_version"] = library_version

        return result

    def _ingest_item(self, item: ZoteroItem) -> None:
        # Use the first PDF attachment
        attachment = next(a for a in item.attachments)
        pdf_bytes = self._zotero.download_pdf(attachment.key)
        text = self._parser.extract_text(pdf_bytes)
        chunks = self._parser.chunk(text, item_key=item.key)

        if self._state.get("ingested", {}).get(item.key) is not None:
            self._vector_store.delete_item(item.key)
            self._metadata_store.delete(item.key)
            if self._lexical_index:
                self._lexical_index.delete_item(item.key)

        self._vector_store.add_chunks(chunks)
        self._metadata_store.save(item)
        if self._lexical_index:
            self._lexical_index.add_paper(item)
            self._lexical_index.add_chunks(chunks)

    def _print_summary(self, result: IngestionResult) -> None:
        parts = [f"[green]{result.ingested} ingested[/green]"]
        if result.skipped_no_pdf:
            parts.append(f"{result.skipped_no_pdf} skipped (no PDF)")
        if result.skipped_already_done:
            parts.append(f"{result.skipped_already_done} already up to date")
        if result.failed:
            parts.append(f"[red]{result.failed} failed[/red]")
        console.print(f"\nDone. {', '.join(parts)}.")

    def _load_state(self) -> dict:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
        return {}

    def _save_state(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self._state, indent=2))
