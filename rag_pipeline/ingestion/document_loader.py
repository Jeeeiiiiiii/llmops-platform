"""
Document Loader — Multi-format document ingestion for the RAG pipeline.

Supports PDF, Markdown, HTML, and plain-text files.  Scans directories
recursively, extracts text content and metadata, and outputs structured
documents as JSON Lines for downstream chunking.

Usage:
    python document_loader.py --source-dir /data/docs --output /data/loaded.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported file extensions and their MIME-type groups
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".text": "text/plain",
    ".rst": "text/plain",
    ".csv": "text/plain",
}


@dataclass
class Document:
    """A loaded document with content and metadata."""

    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF ingestion. Install with: pip install PyMuPDF"
        )

    text_parts: list[str] = []
    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
    return "\n\n".join(text_parts)


def _load_html(path: Path) -> str:
    """Extract visible text from an HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for HTML ingestion. "
            "Install with: pip install beautifulsoup4 lxml"
        )

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    # Remove non-visible elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def _load_markdown(path: Path) -> str:
    """Load Markdown as plain text (preserving structure)."""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_text(path: Path) -> str:
    """Load a plain-text file."""
    return path.read_text(encoding="utf-8", errors="replace")


# Dispatch table
_LOADERS: dict[str, callable] = {
    "application/pdf": _load_pdf,
    "text/markdown": _load_markdown,
    "text/html": _load_html,
    "text/plain": _load_text,
}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _file_metadata(path: Path) -> dict:
    """Extract filesystem-level metadata."""
    stat = path.stat()
    return {
        "source": str(path.resolve()),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _content_hash(content: str) -> str:
    """SHA-256 hash of the document content (for deduplication)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def load_document(path: Path) -> Document | None:
    """
    Load a single document from *path*.

    Returns ``None`` if the file type is unsupported or the file is empty.
    """
    ext = path.suffix.lower()
    mime = SUPPORTED_EXTENSIONS.get(ext)
    if mime is None:
        logger.debug("Skipping unsupported file: %s", path)
        return None

    loader = _LOADERS[mime]
    try:
        content = loader(path)
    except Exception:
        logger.exception("Failed to load %s", path)
        return None

    content = content.strip()
    if not content:
        logger.warning("Empty content after extraction: %s", path)
        return None

    metadata = _file_metadata(path)
    metadata["content_hash"] = _content_hash(content)
    metadata["char_count"] = len(content)
    metadata["mime_type"] = mime

    doc_id = f"doc_{metadata['content_hash']}"
    return Document(doc_id=doc_id, content=content, metadata=metadata)


def scan_directory(source_dir: Path) -> Generator[Document, None, None]:
    """
    Recursively scan *source_dir* and yield loaded ``Document`` objects.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    file_count = 0
    loaded_count = 0

    for root, _dirs, files in os.walk(source_dir):
        for fname in sorted(files):
            file_path = Path(root) / fname
            file_count += 1

            doc = load_document(file_path)
            if doc is not None:
                loaded_count += 1
                yield doc

    logger.info(
        "Directory scan complete: %d files found, %d documents loaded.",
        file_count,
        loaded_count,
    )


def save_documents(documents: Generator[Document, None, None] | list[Document],
                    output_path: Path) -> int:
    """
    Write documents as JSON Lines to *output_path*.

    Returns the number of documents written.
    """
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for doc in documents:
            line = json.dumps(asdict(doc), ensure_ascii=False)
            fh.write(line + "\n")
            count += 1

    logger.info("Wrote %d documents to %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load documents from a directory for the RAG ingestion pipeline."
    )
    parser.add_argument(
        "--source-dir", type=Path, required=True,
        help="Root directory to scan for documents.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for the JSON Lines file.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    docs = scan_directory(args.source_dir)
    written = save_documents(docs, args.output)

    if written == 0:
        logger.warning("No documents were loaded. Check the source directory.")
        sys.exit(1)

    logger.info("Ingestion complete — %d documents ready for chunking.", written)


if __name__ == "__main__":
    main()
