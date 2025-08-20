from __future__ import annotations
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np
from ..chunking import Chunk


def ensure_collection_dense(client: QdrantClient, name: str, dim: int, drop: bool = False):
    """Ensure a dense-vector collection exists with cosine distance.

    Creates (or optionally recreates) a collection configured for a single
    unnamed vector of size `dim` using cosine distance—appropriate for
    L2-normalized embeddings.

    Args:
        client: Initialized Qdrant client.
        name: Collection name.
        dim: Embedding dimensionality (D).
        drop: If True and the collection exists, delete it before creating.

    Notes:
        - If the collection already exists and `drop=False`, this function
          leaves it as-is (no schema validation).
        - Cosine distance is recommended when you L2-normalize vectors
          (dot product becomes equivalent to cosine similarity).
    """
    if drop and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )


def upsert_dense(client: QdrantClient, collection: str, chunks: List[Chunk],
                 vectors: np.ndarray, start_index: int = 0):
    """Upsert points (vector + payload) for the given chunks.

    Each chunk is stored as a single point with:

      - `id`: chunk.id (must be an unsigned integer or a UUID string)
      - `vector`: vectors[j] (list of floats)
      - `payload`: chunk.metadata plus the raw `text` (for RAG)

    Args:
        client: Qdrant client.
        collection: Target collection name.
        chunks: Chunks to write. The order must match `vectors`.
        vectors: 2D NumPy array of shape (N, D), one row per chunk.
        start_index: Reserved for future use (e.g., paging); currently unused.

    Raises:
        ValueError: If the number of chunks and vectors differs.
        qdrant_client.http.exceptions.UnexpectedResponse: If Qdrant rejects the upsert.

    Notes:
        - Qdrant point IDs must be either an unsigned integer or a UUID string.
          Ensure your `Chunk.id` respects that (we recommend UUIDs).
        - The raw chunk text is stored in payload as `"text"`, enabling direct
          answer assembly or debugging without an extra store.
    """
    points = []
    for j, c in enumerate(chunks):
        points.append(PointStruct(
            id=c.id,
            vector=vectors[j].tolist(),
            payload=c.metadata | {"text": c.text}
        ))
    client.upsert(collection_name=collection, points=points)


def search_dense(client: QdrantClient, collection: str, query_vec: list[float], top_k: int = 5,
                 source: str | None = None):
    """Search nearest chunks by vector, optionally filtering by source. For testing.

    Args:
        client: Qdrant client.
        collection: Collection name.
        query_vec: Query embedding as a list (or 1D array) of floats.
        top_k: Maximum number of results to return.
        source: Optional payload filter on `payload['source']` (e.g., "utilitr").

    Returns:
        A list of Qdrant `ScoredPoint` objects with `.score` and `.payload`.

    Example:
        >>> hits = search_dense(client, "utilitr_v1", qvec, top_k=5, source="utilitr")
        >>> urls = [h.payload.get("url") for h in hits]
    """
    flt = None
    if source:
        flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
    hits = client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=flt
    )
    return hits
