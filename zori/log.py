import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import structlog


def configure_logging(debug: bool = False) -> None:
    """Configure structlog + stdlib logging for Zori.

    debug=False  → WARNING+ only, plain text to stderr
    debug=True   → DEBUG+, human-readable to stderr; JSON to timestamped file in LOG_DIR if set
    """
    log_level = logging.DEBUG if debug else logging.WARNING
    log_file = None
    if debug and (log_dir := os.environ.get("LOG_DIR")):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_path / f"{timestamp}.log"

    shared_processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(console_formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(stream_handler)

    if log_file:
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    for noisy in ("httpcore", "httpx", "urllib3", "chromadb", "pyzotero", "openai", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
