"""
Embedding Generator — Encode text chunks and upload to Qdrant.

Uses sentence-transformers (all-MiniLM-L6-v2) to generate dense vector
embeddings from chunked documents, then upserts vectors + payload into
a Qdrant collection in configurable batches.

Usage:
    python embedder.py --input-dir /data/chunks \
                       --qdrant-url http://localhost:6333 \
                       --collection-name documents \
                       --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_QDRANT_URL = "http://localhost:6333"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """A chunk loaded from JSON Lines input."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int = 0
    char_offset: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chunk loader
# ---------------------------------------------------------------------------

def load_chunks_from_dir(input_dir: Path) -> list[ChunkRecord]:
    """
    Load chunk records from all ``*.jsonl`` files in *input_dir*.

    Each line in a JSONL file is expected to have at least ``chunk_id``,
    ``doc_id``, and ``content`` fields.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    chunks: list[ChunkRecord] = []
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        logger.warning("No .jsonl files found in %s", input_dir)
        return chunks

    for jsonl_path in jsonl_files:
        logger.info("Loading chunks from %s", jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d", jsonl_path.name, line_no
                    )
                    continue

                chunk = ChunkRecord(
                    chunk_id=data.get("chunk_id", str(uuid.uuid4())),
                    doc_id=data.get("doc_id", "unknown"),
                    content=data.get("content", ""),
                    chunk_index=data.get("chunk_index", 0),
                    char_offset=data.get("char_offset", 0),
                    metadata=data.get("metadata", {}),
                )
                if chunk.content.strip():
                    chunks.append(chunk)

    logger.info("Loaded %d chunks from %d files", len(chunks), len(jsonl_files))
    return chunks


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(
    chunks: list[ChunkRecord],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of chunks using sentence-transformers.

    Returns an (N, D) numpy array of float32 vectors.
    """
    logger.info(
        "Loading sentence-transformer model: %s", model_name
    )
    model = SentenceTransformer(model_name)

    texts = [chunk.content for chunk in chunks]
    logger.info(
        "Generating embeddings for %d chunks (batch_size=%d)", len(texts), batch_size
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    logger.info(
        "Embeddings generated: shape=%s, dtype=%s",
        embeddings.shape,
        embeddings.dtype,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Qdrant operations
# ---------------------------------------------------------------------------

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = DEFAULT_EMBEDDING_DIM,
) -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Uses cosine distance for normalized embeddings.
    """
    try:
        info = client.get_collection(collection_name)
        logger.info(
            "Collection '%s' already exists (%d points)",
            collection_name,
            info.points_count,
        )
    except (UnexpectedResponse, Exception):
        logger.info(
            "Creating collection '%s' (dim=%d, distance=Cosine)",
            collection_name,
            vector_size,
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def upload_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> int:
    """
    Upsert embeddings, chunk text, and metadata into Qdrant in batches.

    Returns the total number of points upserted.
    """
    total = len(chunks)
    if total == 0:
        logger.warning("No chunks to upload.")
        return 0

    num_batches = (total + batch_size - 1) // batch_size
    upserted = 0

    iterator = range(0, total, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Uploading to Qdrant", total=num_batches)

    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, total)
        batch_chunks = chunks[start_idx:end_idx]
        batch_vectors = embeddings[start_idx:end_idx]

        points = []
        for chunk, vector in zip(batch_chunks, batch_vectors):
            payload = {
                "text": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "char_offset": chunk.char_offset,
                **chunk.metadata,
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        upserted += len(points)

    logger.info("Upserted %d points to collection '%s'", upserted, collection_name)
    return upserted


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_embedding_pipeline(
    input_dir: Path,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> dict:
    """
    End-to-end pipeline: load chunks, embed, and upload to Qdrant.

    Returns a summary dict with counts and status.
    """
    # Load chunks
    chunks = load_chunks_from_dir(input_dir)
    if not chunks:
        return {"status": "no_data", "chunks_loaded": 0, "points_upserted": 0}

    # Generate embeddings
    embeddings = generate_embeddings(
        chunks,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    # Connect to Qdrant and upload
    client = QdrantClient(url=qdrant_url, timeout=60)
    ensure_collection(client, collection_name, vector_size=embeddings.shape[1])
    upserted = upload_to_qdrant(
        client,
        collection_name,
        chunks,
        embeddings,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    return {
        "status": "success",
        "chunks_loaded": len(chunks),
        "points_upserted": upserted,
        "embedding_dim": int(embeddings.shape[1]),
        "model": model_name,
        "collection": collection_name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings from chunked documents and upload to Qdrant."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .jsonl chunk files (from chunker).",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=DEFAULT_QDRANT_URL,
        help=f"Qdrant server URL (default: {DEFAULT_QDRANT_URL}).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Sentence-transformer model (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for encoding and upload (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = run_embedding_pipeline(
        input_dir=args.input_dir,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    if result["status"] == "no_data":
        logger.warning("No chunks found in %s — nothing to embed.", args.input_dir)
        sys.exit(1)

    logger.info(
        "Embedding pipeline complete: %d chunks -> %d points in '%s'",
        result["chunks_loaded"],
        result["points_upserted"],
        result["collection"],
    )


if __name__ == "__main__":
    main()
