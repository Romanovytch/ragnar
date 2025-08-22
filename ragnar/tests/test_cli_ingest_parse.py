from __future__ import annotations
from ragnar.cli.ingest import build_parser, _pick_single_source


def test_cli_parse_minimum_ok():
    p = build_parser()
    ns = p.parse_args([
        "--collection", "utilitr_v1",
        "--qdrant-url", "http://localhost:6333",
        "--embed-api-base", "http://emb/v1",
        "--embed-model", "BAAI/bge-multilingual-gemma2",
    ])
    assert ns.sources_config_path == "sources.yaml"
    assert ns.collection == "utilitr_v1"
    assert ns.source is None


def test_pick_single_source_falls_back_to_only_one():
    resolved = {"utilitr": object()}
    name, cfg = _pick_single_source(resolved, requested=None)
    assert name == "utilitr"


def test_pick_single_source_requires_flag_when_multiple():
    resolved = {"a": object(), "b": object()}
    try:
        _pick_single_source(resolved, requested=None)
        assert False, "should have raised"
    except SystemExit as e:
        assert "Multiple sources" in str(e)


def test_pick_single_source_handles_unknown_requested():
    resolved = {"a": object()}
    try:
        _pick_single_source(resolved, requested="nope")
        assert False, "should have raised"
    except SystemExit as e:
        assert "Unknown source" in str(e)
