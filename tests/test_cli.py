from unittest.mock import MagicMock, patch

from zori.cli import _init_services


def test_init_services_passes_chunk_settings_to_parser():
    config = MagicMock()
    config.ingestion.chunk_size = 500
    config.ingestion.chunk_overlap = 50
    with patch("zori.cli.load_config", return_value=config), \
         patch("zori.cli.ZoteroClient"), \
         patch("zori.cli.get_embeddings"), \
         patch("zori.cli.create_vector_store"), \
         patch("zori.cli.MetadataStore"), \
         patch("zori.cli.LexicalIndex"), \
         patch("zori.cli.PDFParser") as MockParser:
        services, _ = _init_services()

    MockParser.assert_called_once_with(500, 50)
    assert services.pipeline._parser is MockParser.return_value
