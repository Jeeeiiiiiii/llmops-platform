"""
Document Chunker — Split documents into overlapping chunks for embedding.

Implements recursive text splitting that respects sentence and paragraph
boundaries.  Produces chunks with stable IDs, inherited metadata, and
positional information for downstream traceability.

Usage:
    python chunker.py --input /data/loaded.jsonl --output /data/chunks.jsonl \
                      --chunk-size 512 --overlap 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Separator hierarchy — from most to least structural
# ---------------------------------------------------------------------------
_SEPARATORS: list[str] = [
    "\n\n\n",   # major section breaks
    "\n\n",     # paragraph breaks
    "\n",       # line breaks
    ". ",       # sentence endings
    "! ",
    "? ",
    "; ",
    ", ",
    " ",        # word boundaries (last resort)
]


@dataclass
class Chunk:
    """A text chunk produced from a source document."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    char_offset: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recursive splitter
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_size: int, overlap: int,
                separators: list[str] | None = None) -> list[str]:
    """
    Recursively split *text* into chunks of at most *chunk_size* characters
    with *overlap* characters of context between consecutive chunks.

    The splitter tries the most structural separator first (paragraph break)
    and falls through to finer-grained separators when segments are still
    too large.
    """
    if separators is None:
        separators = list(_SEPARATORS)

    # Base case: text fits in a single chunk
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Pick the first separator that actually appears in the text
    sep = ""
    for candidate in separators:
        if candidate in text:
            sep = candidate
            break

    # Split on the chosen separator
    if sep:
        segments = text.split(sep)
    else:
        # No separator found — hard-cut at chunk_size
        segments = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        return [s for s in segments if s.strip()]

    # Merge small segments back together up to chunk_size
    remaining_separators = separators[separators.index(sep) + 1:] if sep in separators else []
    chunks: list[str] = []
    current = ""

    for segment in segments:
        candidate = (current + sep + segment).strip() if current else segment.strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush current buffer
            if current.strip():
                chunks.append(current.strip())
            # If the segment itself exceeds chunk_size, recurse with finer seps
            if len(segment) > chunk_size and remaining_separators:
                sub_chunks = _split_text(segment, chunk_size, overlap, remaining_separators)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = segment.strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Add *overlap* characters from the end of the previous chunk to the
    beginning of each subsequent chunk so that context is not lost at
    boundaries.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Take up to `overlap` chars from the end of the previous chunk
        overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
        # Find a clean word boundary in the overlap prefix
        space_idx = overlap_text.find(" ")
        if space_idx > 0:
            overlap_text = overlap_text[space_idx + 1:]
        result.append(overlap_text + " " + chunks[i])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(doc_id: str, content: str, metadata: dict,
                   chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """
    Split *content* into chunks, returning ``Chunk`` objects with inherited
    metadata and positional information.
    """
    raw_chunks = _split_text(content, chunk_size, overlap)
    overlapped = _add_overlap(raw_chunks, overlap)

    chunks: list[Chunk] = []
    char_offset = 0

    for idx, text in enumerate(overlapped):
        # Deterministic chunk ID from doc + index
        chunk_hash = hashlib.sha256(f"{doc_id}:{idx}".encode()).hexdigest()[:12]
        chunk_id = f"chunk_{chunk_hash}"

        chunk_meta = {
            **metadata,
            "chunk_size": len(text),
            "chunk_index": idx,
            "total_chunks": len(overlapped),
        }

        chunks.append(Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=text,
            chunk_index=idx,
            char_offset=char_offset,
            metadata=chunk_meta,
        ))
        char_offset += len(text)

    return chunks


def process_jsonl(input_path: Path, output_path: Path,
                  chunk_size: int = 512, overlap: int = 50) -> int:
    """
    Read documents from *input_path* (JSON Lines), chunk each, and write
    chunks to *output_path*.  Returns the total number of chunks produced.
    """
    total_chunks = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_no)
                continue

            doc_id = doc.get("doc_id", f"unknown_{line_no}")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if not content.strip():
                logger.warning("Empty document at line %d (doc_id=%s)", line_no, doc_id)
                continue

            chunks = chunk_document(doc_id, content, metadata, chunk_size, overlap)
            for chunk in chunks:
                fout.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            logger.debug("doc_id=%s -> %d chunks", doc_id, len(chunks))

    logger.info("Chunking complete: %d total chunks from %s", total_chunks, input_path)
    return total_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk documents for the RAG embedding pipeline."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Input JSON Lines file (from document_loader).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON Lines file with chunks.")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Maximum chunk size in characters (default: 512).")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap between consecutive chunks (default: 50).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    total = process_jsonl(args.input, args.output, args.chunk_size, args.overlap)
    if total == 0:
        logger.warning("No chunks produced — check input file.")
    else:
        logger.info("Done — %d chunks written to %s", total, args.output)


if __name__ == "__main__":
    main()
