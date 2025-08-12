from __future__ import annotations
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np
from ..chunking import Chunk


def ensure_collection_dense(client: QdrantClient, name: str, dim: int, drop: bool = False):
    if drop and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )


def upsert_dense(client: QdrantClient, collection: str, chunks: List[Chunk],
                 vectors: np.ndarray, start_index: int = 0):
    points = []
    for j, c in enumerate(chunks):
        points.append(PointStruct(
            id=c.id,
            vector=vectors[j].tolist(),
            payload=c.metadata | {"text": c.text}  # store raw text for RAG
        ))
    client.upsert(collection_name=collection, points=points)


def search_dense(client: QdrantClient, collection: str, query_vec: list[float], top_k: int = 5,
                 source: str | None = None):
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
