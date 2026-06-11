from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ..chunking import Chunk
from ..embeddings.named import NamedVectorBatch, SparseVectorData
from ..sources.models.base import (
    DenseVectorConfig,
    SparseVectorConfig,
    VectorIndexConfig,
)
from ..sources.models.base import (
    MultiVectorConfig as AgoraMultiVectorConfig,
)


def _distance(name: str) -> Distance:
    if name == "cosine":
        return Distance.COSINE
    raise ValueError(f"Unsupported vector distance: {name}")


def _sparse_modifier(name: str) -> Modifier | None:
    if name == "idf":
        return Modifier.IDF
    if name == "none":
        return None
    raise ValueError(f"Unsupported sparse vector modifier: {name}")


def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_index: VectorIndexConfig,
    dimensions: dict[str, int],
    drop: bool = False,
):
    """Ensure a Qdrant collection exists with the configured named vectors."""
    if drop and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        vectors_config: dict[str, VectorParams] = {}
        sparse_vectors_config: dict[str, SparseVectorParams] = {}

        for vector in vector_index.vectors:
            if isinstance(vector, DenseVectorConfig):
                vectors_config[vector.name] = VectorParams(
                    size=dimensions[vector.name],
                    distance=_distance(vector.distance),
                )
            elif isinstance(vector, SparseVectorConfig):
                sparse_vectors_config[vector.name] = SparseVectorParams(
                    modifier=_sparse_modifier(vector.modifier)
                )
            elif isinstance(vector, AgoraMultiVectorConfig):
                vectors_config[vector.name] = VectorParams(
                    size=dimensions[vector.name],
                    distance=_distance(vector.distance),
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )

        kwargs = {
            "collection_name": name,
            "vectors_config": vectors_config,
        }
        if sparse_vectors_config:
            kwargs["sparse_vectors_config"] = sparse_vectors_config
        client.create_collection(**kwargs)


def ensure_collection_dense(client: QdrantClient, name: str, dim: int, drop: bool = False):
    """Backward-compatible dense collection helper using named vector 'dense'."""
    ensure_collection(
        client,
        name,
        VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=dim)]),
        {"dense": dim},
        drop=drop,
    )


def validate_named_vectors(
    chunks: list[Chunk],
    vector_index: VectorIndexConfig,
    named_vectors: NamedVectorBatch,
    dimensions: dict[str, int],
) -> None:
    configured_names = {v.name for v in vector_index.vectors}
    returned_names = set(named_vectors)
    if configured_names != returned_names:
        missing = ", ".join(sorted(configured_names - returned_names)) or "none"
        unexpected = ", ".join(sorted(returned_names - configured_names)) or "none"
        raise ValueError(
            f"Vector names do not match config; missing={missing}; unexpected={unexpected}"
        )

    n_chunks = len(chunks)
    for vector in vector_index.vectors:
        values = named_vectors[vector.name]
        if isinstance(vector, DenseVectorConfig):
            if not isinstance(values, np.ndarray) or values.ndim != 2:
                raise ValueError(f"Dense vector '{vector.name}' must be a 2D NumPy array")
            if values.shape != (n_chunks, dimensions[vector.name]):
                raise ValueError(
                    f"Dense vector '{vector.name}' shape {values.shape} does not match "
                    f"({n_chunks}, {dimensions[vector.name]})"
                )
        elif isinstance(vector, SparseVectorConfig):
            if not isinstance(values, list) or len(values) != n_chunks:
                raise ValueError(f"Sparse vector '{vector.name}' must contain one value per chunk")
            for item in values:
                if not isinstance(item, SparseVectorData):
                    raise ValueError(f"Sparse vector '{vector.name}' has an invalid item")
                if len(item.indices) != len(item.values):
                    raise ValueError(
                        f"Sparse vector '{vector.name}' indices and values lengths differ"
                    )
        elif isinstance(vector, AgoraMultiVectorConfig):
            if not isinstance(values, list) or len(values) != n_chunks:
                raise ValueError(f"Multi vector '{vector.name}' must contain one value per chunk")
            expected_dim = dimensions[vector.name]
            for item in values:
                if not isinstance(item, list):
                    raise ValueError(f"Multi vector '{vector.name}' has an invalid item")
                for row in item:
                    if len(row) != expected_dim:
                        raise ValueError(
                            f"Multi vector '{vector.name}' row dimension {len(row)} "
                            f"does not match {expected_dim}"
                        )


def build_named_points(
    chunks: list[Chunk],
    vector_index: VectorIndexConfig,
    named_vectors: NamedVectorBatch,
    dimensions: dict[str, int],
) -> list[PointStruct]:
    validate_named_vectors(chunks, vector_index, named_vectors, dimensions)
    points: list[PointStruct] = []
    for j, chunk in enumerate(chunks):
        point_vectors = {}
        for vector in vector_index.vectors:
            values = named_vectors[vector.name]
            if isinstance(vector, DenseVectorConfig):
                point_vectors[vector.name] = values[j].tolist()
            elif isinstance(vector, SparseVectorConfig):
                sparse = values[j]
                point_vectors[vector.name] = SparseVector(
                    indices=sparse.indices,
                    values=sparse.values,
                )
            elif isinstance(vector, AgoraMultiVectorConfig):
                point_vectors[vector.name] = values[j]

        points.append(
            PointStruct(
                id=chunk.id,
                vector=point_vectors,
                payload=chunk.metadata | {"text": chunk.text},
            )
        )
    return points


def upsert_named(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    vector_index: VectorIndexConfig,
    named_vectors: NamedVectorBatch,
    dimensions: dict[str, int],
):
    """Upsert named-vector points for the given chunks."""
    points = build_named_points(chunks, vector_index, named_vectors, dimensions)
    client.upsert(collection_name=collection, points=points)


def upsert_dense(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    start_index: int = 0,
):
    """Backward-compatible dense upsert helper using named vector 'dense'."""
    del start_index
    dimensions = {"dense": int(vectors.shape[1]) if vectors.ndim == 2 else 0}
    upsert_named(
        client,
        collection,
        chunks,
        VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=dimensions["dense"])]),
        {"dense": vectors},
        dimensions,
    )


def search_dense(
    client: QdrantClient,
    collection: str,
    query_vec: list[float],
    top_k: int = 5,
    source: str | None = None,
) -> list[ScoredPoint]:
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
    response = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=flt,
    )
    return response.points
