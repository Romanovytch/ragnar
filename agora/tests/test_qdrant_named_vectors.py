from __future__ import annotations

import numpy as np
import pytest
from qdrant_client.http.models import Modifier, ScoredPoint

from agora.chunking import Chunk
from agora.embeddings.named import SparseVectorData
from agora.sources.models.base import (
    DenseVectorConfig,
    MultiVectorConfig,
    SparseVectorConfig,
    VectorIndexConfig,
)
from agora.vectorstores.qdrant_store import (
    build_named_points,
    build_payload_points,
    ensure_collection,
    ensure_payload_collection,
    fetch_parent_chunks_for_hits,
    upsert_named,
    upsert_payload_points,
    validate_named_vectors,
)


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="11111111-1111-5111-8111-111111111111", text="alpha beta", metadata={"n": 1}),
        Chunk(id="22222222-2222-5222-8222-222222222222", text="gamma", metadata={"n": 2}),
    ]


def test_dense_only_point_shape_uses_named_vector_and_preserves_payload():
    chunks = _chunks()
    config = VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=2)])
    vectors = {"dense": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")}

    points = build_named_points(chunks, config, vectors, {"dense": 2})

    assert points[0].id == chunks[0].id
    assert points[0].vector == {"dense": [1.0, 2.0]}
    assert points[0].payload == {"n": 1, "text": "alpha beta"}


def test_point_payload_keeps_raw_text_without_embedding_text():
    chunks = [
        Chunk(
            id="11111111-1111-5111-8111-111111111111",
            text="Run this command.",
            metadata={"breadcrumbs": ["Doc", "Install"]},
        )
    ]
    config = VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=2)])
    vectors = {"dense": np.array([[1.0, 2.0]], dtype="float32")}

    points = build_named_points(chunks, config, vectors, {"dense": 2})

    assert points[0].payload == {
        "breadcrumbs": ["Doc", "Install"],
        "text": "Run this command.",
    }


def test_hybrid_point_shape_uses_same_point_identity_for_all_vectors():
    chunks = _chunks()
    config = VectorIndexConfig(
        vectors=[
            DenseVectorConfig(name="dense", size=2),
            SparseVectorConfig(name="sparse"),
            MultiVectorConfig(name="multi", size=2),
        ]
    )
    vectors = {
        "dense": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        "sparse": [
            SparseVectorData(indices=[1, 3], values=[0.5, 0.7]),
            SparseVectorData(indices=[2], values=[0.9]),
        ],
        "multi": [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
        ],
    }

    points = build_named_points(chunks, config, vectors, {"dense": 2, "multi": 2})

    assert [p.id for p in points] == [c.id for c in chunks]
    assert set(points[0].vector) == {"dense", "sparse", "multi"}
    assert points[0].vector["dense"] == [1.0, 2.0]
    assert points[0].vector["multi"] == [[1.0, 2.0], [3.0, 4.0]]
    assert points[0].vector["sparse"].indices == [1, 3]
    assert points[0].vector["sparse"].values == [0.5, 0.7]
    assert points[0].payload["text"] == chunks[0].text


def test_validation_fails_for_missing_configured_vector():
    config = VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=2)])

    with pytest.raises(ValueError, match="missing=dense"):
        validate_named_vectors(_chunks(), config, {}, {"dense": 2})


def test_validation_fails_for_unexpected_vector():
    config = VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=2)])
    vectors = {
        "dense": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
        "other": np.array([[1.0], [2.0]], dtype="float32"),
    }

    with pytest.raises(ValueError, match="unexpected=other"):
        validate_named_vectors(_chunks(), config, vectors, {"dense": 2})


def test_validation_fails_for_wrong_dense_dimension():
    config = VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=3)])
    vectors = {"dense": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")}

    with pytest.raises(ValueError, match="Dense vector 'dense' shape"):
        validate_named_vectors(_chunks(), config, vectors, {"dense": 3})


def test_validation_fails_for_wrong_multi_dimension():
    config = VectorIndexConfig(vectors=[MultiVectorConfig(name="multi", size=3)])
    vectors = {"multi": [[[1.0, 2.0]], [[3.0, 4.0]]]}

    with pytest.raises(ValueError, match="row dimension 2 does not match 3"):
        validate_named_vectors(_chunks(), config, vectors, {"multi": 3})


def test_validation_fails_for_sparse_length_mismatch():
    config = VectorIndexConfig(vectors=[SparseVectorConfig(name="sparse")])
    vectors = {"sparse": [SparseVectorData(indices=[1], values=[]), SparseVectorData([], [])]}

    with pytest.raises(ValueError, match="indices and values lengths differ"):
        validate_named_vectors(_chunks(), config, vectors, {})


def test_ensure_collection_configures_bm25_sparse_idf_modifier():
    class FakeClient:
        kwargs = None

        def collection_exists(self, name):
            assert name == "kb"
            return False

        def create_collection(self, **kwargs):
            self.kwargs = kwargs

    client = FakeClient()
    config = VectorIndexConfig(vectors=[SparseVectorConfig(name="sparse")])

    ensure_collection(client, "kb", config, dimensions={})

    sparse_config = client.kwargs["sparse_vectors_config"]["sparse"]
    assert sparse_config.modifier == Modifier.IDF


def test_upsert_named_uses_batched_upload_points():
    class FakeClient:
        kwargs = None

        def upload_points(self, **kwargs):
            self.kwargs = kwargs

    client = FakeClient()
    chunks = _chunks()
    config = VectorIndexConfig(vectors=[DenseVectorConfig(name="dense", size=2)])
    vectors = {"dense": np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")}

    upsert_named(
        client,
        "kb",
        chunks,
        config,
        vectors,
        {"dense": 2},
        batch_size=3,
    )

    assert client.kwargs["collection_name"] == "kb"
    assert client.kwargs["batch_size"] == 3
    assert len(client.kwargs["points"]) == 2


def test_payload_points_store_parent_chunk_id_and_text_without_vectors():
    chunks = [Chunk(id="33333333-3333-5333-8333-333333333333", text="parent", metadata={"n": 3})]

    points = build_payload_points(chunks)

    assert points[0].id == chunks[0].id
    assert points[0].vector == {}
    assert points[0].payload == {
        "n": 3,
        "chunk_id": chunks[0].id,
        "text": "parent",
    }


def test_ensure_payload_collection_creates_vectorless_collection():
    class FakeClient:
        kwargs = None

        def collection_exists(self, name):
            assert name == "kb_parents"
            return False

        def create_collection(self, **kwargs):
            self.kwargs = kwargs

    client = FakeClient()

    ensure_payload_collection(client, "kb_parents")

    assert client.kwargs == {"collection_name": "kb_parents", "vectors_config": {}}


def test_upsert_payload_points_uses_batched_upload_points():
    class FakeClient:
        kwargs = None

        def upload_points(self, **kwargs):
            self.kwargs = kwargs

    client = FakeClient()
    chunks = [Chunk(id="33333333-3333-5333-8333-333333333333", text="parent", metadata={})]

    upsert_payload_points(client, "kb_parents", chunks, batch_size=5)

    assert client.kwargs["collection_name"] == "kb_parents"
    assert client.kwargs["batch_size"] == 5
    assert len(client.kwargs["points"]) == 1


def test_fetch_parent_chunks_for_hits_deduplicates_parent_ids():
    class FakeClient:
        kwargs = None

        def retrieve(self, **kwargs):
            self.kwargs = kwargs
            return ["parent"]

    client = FakeClient()
    hits = [
        ScoredPoint(id="1", version=0, score=0.9, payload={"parent_id": "p1"}),
        ScoredPoint(id="2", version=0, score=0.8, payload={"parent_id": "p1"}),
        ScoredPoint(id="3", version=0, score=0.7, payload={"parent_id": "p2"}),
    ]

    parents = fetch_parent_chunks_for_hits(client, "kb_parents", hits)

    assert parents == ["parent"]
    assert client.kwargs == {
        "collection_name": "kb_parents",
        "ids": ["p1", "p2"],
        "with_payload": True,
        "with_vectors": False,
    }
