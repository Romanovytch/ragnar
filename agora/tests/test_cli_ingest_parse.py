from __future__ import annotations

from types import SimpleNamespace

import pytest

from agora.cli.ingest import (
    _chunk_metadata,
    _drop_parent_collection_if_requested,
    _parent_collection_name,
    _pick_single_source,
    _validate_parent_storage_config,
    build_parser,
)


def test_cli_parse_minimum_ok():
    p = build_parser()
    ns = p.parse_args(
        [
            "--collection",
            "utilitr_v1",
            "--qdrant-url",
            "http://localhost:6333",
            "--embed-api-base",
            "http://emb/v1",
            "--embed-model",
            "BAAI/bge-multilingual-gemma2",
        ]
    )
    assert ns.sources_config_path == "sources.yaml"
    assert ns.collection == "utilitr_v1"
    assert ns.source is None
    assert ns.qdrant_batch_size == 16


def test_pick_single_source_falls_back_to_only_one():
    resolved = {"utilitr": object()}
    name, cfg = _pick_single_source(resolved, requested=None)
    assert name == "utilitr"


def test_pick_single_source_requires_flag_when_multiple():
    resolved = {"a": object(), "b": object()}
    try:
        _pick_single_source(resolved, requested=None)
        AssertionError(), "should have raised"
    except SystemExit as e:
        assert "Multiple sources" in str(e)


def test_pick_single_source_handles_unknown_requested():
    resolved = {"a": object()}
    try:
        _pick_single_source(resolved, requested="nope")
        AssertionError(), "should have raised"
    except SystemExit as e:
        assert "Unknown source" in str(e)


def test_parent_storage_size_validation_requires_larger_parent_budgets():
    cfg = SimpleNamespace(
        parent_storage_mode="classic",
        parent_target_tokens=800,
        parent_max_tokens=1200,
    )
    args = SimpleNamespace(target_tokens=800, max_tokens=1200)

    with pytest.raises(SystemExit, match="parent_target_tokens"):
        _validate_parent_storage_config(cfg, args)


def test_parent_storage_size_validation_requires_larger_parent_max_budget():
    cfg = SimpleNamespace(
        parent_storage_mode="classic",
        parent_target_tokens=900,
        parent_max_tokens=1200,
    )
    args = SimpleNamespace(target_tokens=800, max_tokens=1200)

    with pytest.raises(SystemExit, match="parent_max_tokens"):
        _validate_parent_storage_config(cfg, args)


def test_parent_storage_size_validation_is_skipped_when_disabled():
    cfg = SimpleNamespace(
        parent_storage_mode="none",
        parent_target_tokens=1,
        parent_max_tokens=1,
    )
    args = SimpleNamespace(target_tokens=800, max_tokens=1200)

    _validate_parent_storage_config(cfg, args)


def test_parent_metadata_uses_parent_chunk_index_without_child_chunk_index():
    meta = _chunk_metadata(
        {"doc_title": "Doc", "source_url": "https://example.org/doc"},
        "Parent text",
        [(1, "Doc"), (2, "Section")],
        3,
        index_key="parent_chunk_index",
    )

    assert meta["parent_chunk_index"] == 3
    assert "chunk_index" not in meta


def test_drop_parent_collection_if_requested_uses_parent_collection_name():
    class FakeClient:
        deleted = []

        def collection_exists(self, name):
            return name == "kb_parents"

        def delete_collection(self, name):
            self.deleted.append(name)

    client = FakeClient()

    _drop_parent_collection_if_requested(client, "kb", drop=True)

    assert _parent_collection_name("kb") == "kb_parents"
    assert client.deleted == ["kb_parents"]


def test_drop_parent_collection_if_requested_skips_without_drop_flag():
    class FakeClient:
        deleted = []

        def collection_exists(self, name):
            return True

        def delete_collection(self, name):
            self.deleted.append(name)

    client = FakeClient()

    _drop_parent_collection_if_requested(client, "kb", drop=False)

    assert client.deleted == []
