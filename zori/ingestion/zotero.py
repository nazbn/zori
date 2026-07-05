from dataclasses import dataclass, field

from pyzotero import zotero


@dataclass
class ZoteroCollection:
    key: str
    name: str
    parent_key: str | None = None
    num_items: int = 0


@dataclass
class ZoteroAttachment:
    key: str
    filename: str
    parent_key: str
    md5: str | None = None


@dataclass
class ZoteroItem:
    key: str
    title: str
    version: int = 0
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    abstract: str | None = None
    tags: list[str] = field(default_factory=list)
    journal: str | None = None
    doi: str | None = None
    item_type: str = "journalArticle"
    attachments: list[ZoteroAttachment] = field(default_factory=list)


class ZoteroClient:
    def __init__(self, library_id: str, library_type: str, api_key: str):
        self._zot = zotero.Zotero(library_id, library_type, api_key)

    def fetch_collections(self) -> list[ZoteroCollection]:
        """Return all top-level and nested collections in the library."""
        raw = self._zot.everything(self._zot.collections())
        return [
            ZoteroCollection(
                key=c["key"],
                name=c["data"]["name"],
                parent_key=c["data"].get("parentCollection") or None,
                num_items=c["meta"].get("numItems", 0),
            )
            for c in raw
        ]

    def find_collection(self, name: str) -> ZoteroCollection | None:
        """Find a collection by name or path (case-insensitive).
        Path syntax: "Parent/Child/Grandchild" walks the hierarchy level by level.
        Plain name matches any collection regardless of depth (first match wins)."""
        cols = self.fetch_collections()
        parts = [p.strip() for p in name.split("/")]

        if len(parts) == 1:
            name_lower = name.lower()
            for col in cols:
                if col.name.lower() == name_lower:
                    return col
            return None

        # Walk hierarchy level by level
        parent_key: str | None = None
        current: ZoteroCollection | None = None
        for part in parts:
            part_lower = part.lower()
            current = next(
                (c for c in cols if c.name.lower() == part_lower and c.parent_key == parent_key),
                None,
            )
            if current is None:
                return None
            parent_key = current.key
        return current

    def fetch_all_items(self) -> list[ZoteroItem]:
        """Fetch the entire library in one paginated call."""
        raw = self._zot.everything(self._zot.items())
        return self._parse_items(raw)

    def fetch_collection_items(self, collection_key: str) -> list[ZoteroItem]:
        """Fetch all items in a specific collection."""
        raw = self._zot.everything(self._zot.collection_items(collection_key))
        return self._parse_items(raw)

    def fetch_new_items(self, since_version: int) -> list[ZoteroItem]:
        """Fetch only items added or modified since the given library version."""
        raw = self._zot.everything(self._zot.items(since=since_version))
        return self._parse_items(raw)

    def get_library_version(self) -> int:
        """Return the current library version number for sync tracking."""
        self._zot.items(limit=1)
        return int(self._zot.request.headers.get("Last-Modified-Version", 0))

    def write_note(self, item_key: str, html: str, tags: list[str] | None = None) -> None:
        """Save an HTML note as a child of the given item in Zotero."""
        self._zot.create_items([{
            "itemType": "note",
            "parentItem": item_key,
            "note": html,
            "tags": [{"tag": t} for t in (tags or [])],
        }])

    def download_pdf(self, attachment_key: str) -> bytes:
        """Download a PDF attachment and return its raw bytes."""
        return self._zot.file(attachment_key)

    def _parse_items(self, raw_items: list[dict]) -> list[ZoteroItem]:
        papers: dict[str, ZoteroItem] = {}
        pdf_attachments: list[dict] = []

        for raw in raw_items:
            data = raw["data"]
            item_type = data.get("itemType")

            if item_type == "attachment":
                if data.get("contentType") == "application/pdf":
                    pdf_attachments.append(data)
            elif item_type not in ("note", "annotation"):
                papers[raw["key"]] = self._parse_paper(raw)

        for att_data in pdf_attachments:
            attachment = ZoteroAttachment(
                key=att_data["key"],
                filename=att_data.get("filename", ""),
                parent_key=att_data.get("parentItem") or att_data["key"],
                md5=att_data.get("md5"),
            )
            parent_key = att_data.get("parentItem")

            if parent_key and parent_key in papers:
                papers[parent_key].attachments.append(attachment)
            elif not parent_key:
                # Top-level attachment — wrap in a minimal ZoteroItem
                papers[att_data["key"]] = ZoteroItem(
                    key=att_data["key"],
                    title=att_data.get("title") or att_data.get("filename", "Untitled"),
                    item_type="attachment",
                    attachments=[attachment],
                )

        return list(papers.values())

    def _parse_paper(self, raw: dict) -> ZoteroItem:
        data = raw["data"]

        authors = []
        for creator in data.get("creators", []):
            if creator.get("creatorType") == "author":
                name = creator.get("name")
                if not name:
                    first = creator.get("firstName", "")
                    last = creator.get("lastName", "")
                    name = f"{first} {last}".strip()
                if name:
                    authors.append(name)

        date_str = data.get("date", "")
        year = date_str[:4] if date_str else None

        return ZoteroItem(
            key=raw["key"],
            title=data.get("title", "Untitled"),
            version=raw.get("version", 0),
            authors=authors,
            year=year,
            abstract=data.get("abstractNote") or None,
            tags=[t["tag"] for t in data.get("tags", [])],
            journal=data.get("publicationTitle") or None,
            doi=data.get("DOI") or None,
            item_type=data.get("itemType", "journalArticle"),
        )
