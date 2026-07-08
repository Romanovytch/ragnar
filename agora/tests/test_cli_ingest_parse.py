from __future__ import annotations

from types import SimpleNamespace

import pytest

from agora.chunking import Chunk
from agora.cli.ingest import (
    _chunk_metadata,
    _drop_parent_collection_if_requested,
    _embedding_text,
    _parent_collection_name,
    _pick_single_source,
    _summary_collection_name,
    _validate_parent_storage_config,
    _validate_summary_config,
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
    assert ns.split_heading_level is None


def test_cli_parse_accepts_split_heading_level():
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
            "--split-heading-level",
            "3",
        ]
    )

    assert ns.split_heading_level == 3


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


def test_embedding_text_prefixes_breadcrumbs_without_changing_raw_chunk_text():
    chunk = Chunk(
        id="c1",
        text="Run this command.",
        metadata={"breadcrumbs": ["Doc", "Install"]},
    )

    text = _embedding_text(chunk)

    assert text == "Doc > Install\n\nRun this command."
    assert chunk.text == "Run this command."
    assert "embedding_text" not in chunk.metadata


def test_embedding_text_falls_back_to_doc_title_when_breadcrumbs_are_missing():
    chunk = Chunk(
        id="c1",
        text="Install the package.",
        metadata={"doc_title": "Setup"},
    )

    assert _embedding_text(chunk) == "Setup\n\nInstall the package."


def test_embedding_text_returns_raw_text_without_heading_context():
    chunk = Chunk(id="c1", text="Plain text.", metadata={})

    assert _embedding_text(chunk) == "Plain text."


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


def test_summary_config_validation_is_skipped_when_disabled():
    cfg = SimpleNamespace(
        synthetic_llm_summary=False,
        synthetic_llm_summary_group_max_tokens=1,
        synthetic_llm_summary_max_sentences=1,
    )
    args = SimpleNamespace(max_tokens=1200)

    _validate_summary_config(cfg, args)


def test_summary_config_validation_requires_larger_group_budget():
    cfg = SimpleNamespace(
        synthetic_llm_summary=True,
        synthetic_llm_summary_group_max_tokens=1200,
        synthetic_llm_summary_max_sentences=3,
    )
    args = SimpleNamespace(max_tokens=1200)

    with pytest.raises(SystemExit, match="synthetic_llm_summary_group_max_tokens"):
        _validate_summary_config(cfg, args)


def test_summary_collection_name_uses_sibling_collection():
    assert _summary_collection_name("kb") == "kb_summaries"
