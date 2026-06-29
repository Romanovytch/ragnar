# tests/test_sources_loader.py
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from agora.sources.loader import load_sources_config, validate_and_resolve
from agora.sources.models.base import (
    DenseVectorConfig,
    MultiVectorConfig,
    SparseVectorConfig,
)
from agora.sources.models.markdown_repo import MarkdownRepoConfig


def _write_yaml(p: Path, s: str) -> None:
    p.write_text(textwrap.dedent(s).lstrip(), encoding="utf-8")


def test_validate_and_resolve_minimal_markdown_source_ok(tmp_path: Path):
    repo = tmp_path / "utilitR"
    repo.mkdir()
    (repo / "chap.md").write_text("# Title\n\nBody\n", encoding="utf-8")

    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        defaults:
          include_globs: ["**/*.md", "**/*.qmd", "**/*.Rmd"]
          exclude_dirs: [".git"]
          default_lang: "en"
          follow_symlinks: false
        sources:
          utilitr:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://book.utilitr.org"
            repo_url_template: "https://github.com/InseeFrLab/utilitR/blob/{{commit}}/{{path}}"
        """,
    )

    resolved = validate_and_resolve(cfg)
    cfg_util = resolved["utilitr"]
    assert isinstance(cfg_util, MarkdownRepoConfig)
    assert cfg_util.kind == "markdown_repo"
    # base_url is normalized with trailing slash
    assert cfg_util.base_url.endswith("/")
    # defaults merged
    assert cfg_util.include_globs == ["**/*.md", "**/*.qmd", "**/*.Rmd"]
    assert cfg_util.exclude_dirs == [".git"]
    assert cfg_util.default_lang == "en"
    # repo_path resolved absolute
    assert cfg_util.repo_path.is_absolute()
    assert cfg_util.repo_path == repo.resolve()


def test_override_defaults_preserved(tmp_path: Path):
    repo = tmp_path / "handbook"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        defaults:
          include_globs: ["**/*.md", "**/*.qmd"]
          exclude_dirs: [".git", "build"]
          default_lang: "en"
        sources:
          hb:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
            repo_url_template: "https://github.com/Org/handbook/blob/{{commit}}/{{path}}"
            include_globs: ["**/*.md"]         # override
            default_lang: "fr"                 # override
        """,
    )
    resolved = validate_and_resolve(cfg)
    hb = resolved["hb"]
    assert hb.include_globs == ["**/*.md"]  # user value kept
    assert hb.default_lang == "fr"  # user value kept
    assert hb.exclude_dirs == [".git", "build"]  # default came through


def test_repo_url_template_validation_fails(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        sources:
          bad:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
            repo_url_template: "https://github.com/Org/repo/blob/main/NO_PATH_PLACEHOLDER"
        """,
    )
    with pytest.raises(ValidationError):
        validate_and_resolve(cfg)


def test_repo_path_must_exist(tmp_path: Path):
    cfg = tmp_path / "sources.yaml"
    non_existent = tmp_path / "missing"
    _write_yaml(
        cfg,
        f"""
        version: 1
        sources:
          s:
            kind: markdown_repo
            repo_path: "{non_existent}"
            base_url: "https://docs.example.org"
            repo_url_template: "https://github.com/Org/repo/blob/{{commit}}/{{path}}"
        """,
    )
    with pytest.raises(ValidationError):
        validate_and_resolve(cfg)


def test_relative_repo_path_resolves_against_yaml_dir(tmp_path: Path):
    # Tree: tmp/conf/sources.yaml  and tmp/repo/...
    repo = tmp_path / "repo"
    repo.mkdir()
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    cfg = conf_dir / "sources.yaml"
    _write_yaml(
        cfg,
        """
        version: 1
        sources:
          s:
            kind: markdown_repo
            repo_path: "../repo"
            base_url: "https://docs.example.org"
            repo_url_template: "https://github.com/Org/repo/blob/{commit}/{path}"
        """,
    )
    resolved = validate_and_resolve(cfg)
    s = resolved["s"]
    assert s.repo_path == repo.resolve()


def test_unknown_defaults_keys_are_ignored(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        defaults:
          include_globs: ["**/*.md"]
          default_lang: "en"
          frobnicate: 123    # unknown; should not crash or bleed into config
        sources:
          s:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
            repo_url_template: "https://github.com/Org/repo/blob/{{commit}}/{{path}}"
        """,
    )
    resolved = validate_and_resolve(cfg)
    s = resolved["s"]
    assert s.include_globs == ["**/*.md"]
    assert s.default_lang == "en"
    # ensure the model has no attribute 'frobnicate'
    assert not hasattr(s, "frobnicate")


def test_multiple_sources_resolve_independently(tmp_path: Path):
    repo1 = tmp_path / "a"
    repo1.mkdir()
    repo2 = tmp_path / "b"
    repo2.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        defaults:
          include_globs: ["**/*.md", "**/*.qmd"]
          default_lang: "en"
        sources:
          one:
            kind: markdown_repo
            repo_path: "{repo1}"
            base_url: "https://site.one"
            repo_url_template: "https://git/one/blob/{{commit}}/{{path}}"
            default_lang: "fr"
          two:
            kind: markdown_repo
            repo_path: "{repo2}"
            base_url: "https://site.two/"
            repo_url_template: "https://git/two/blob/{{commit}}/{{path}}"
            include_globs: ["**/*.qmd"]
        """,
    )
    resolved = validate_and_resolve(cfg)
    one, two = resolved["one"], resolved["two"]
    assert one.default_lang == "fr"  # source override
    assert two.include_globs == ["**/*.qmd"]  # source override
    assert one.base_url.endswith("/") and two.base_url.endswith("/")  # normalized


def test_vector_index_defaults_to_named_dense(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        sources:
          s:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
        """,
    )

    loaded = load_sources_config(cfg)

    assert len(loaded.vector_index.vectors) == 1
    dense = loaded.vector_index.vectors[0]
    assert isinstance(dense, DenseVectorConfig)
    assert dense.name == "dense"


def test_vector_index_validates_hybrid_modes(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        vector_index:
          vectors:
            - name: dense
              kind: dense
              size: 3
            - name: sparse
              kind: sparse
            - name: multi
              kind: multi
              size: 2
        sources:
          s:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
        """,
    )

    loaded = load_sources_config(cfg)

    dense, sparse, multi = loaded.vector_index.vectors
    assert isinstance(dense, DenseVectorConfig)
    assert isinstance(sparse, SparseVectorConfig)
    assert isinstance(multi, MultiVectorConfig)
    assert dense.size == 3
    assert sparse.provider == "fastembed"
    assert sparse.model == "Qdrant/bm25"
    assert sparse.modifier == "idf"
    assert sparse.language is None
    assert multi.provider == "fastembed"
    assert multi.model == "answerdotai/answerai-colbert-small-v1"
    assert multi.size == 2


def test_vector_index_multi_defaults_to_fastembed_late_interaction(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        vector_index:
          vectors:
            - name: multi
              kind: multi
        sources:
          s:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
        """,
    )

    loaded = load_sources_config(cfg)

    multi = loaded.vector_index.vectors[0]
    assert isinstance(multi, MultiVectorConfig)
    assert multi.provider == "fastembed"
    assert multi.model == "answerdotai/answerai-colbert-small-v1"
    assert multi.size is None


def test_markdown_source_accepts_classic_parent_storage_config(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = tmp_path / "sources.yaml"
    _write_yaml(
        cfg,
        f"""
        version: 1
        sources:
          s:
            kind: markdown_repo
            repo_path: "{repo}"
            base_url: "https://docs.example.org"
            parent_storage_mode: classic
            parent_target_tokens: 900
            parent_overlap_tokens: 80
            parent_max_tokens: 1200
        """,
    )

    resolved = validate_and_resolve(cfg)
    s = resolved["s"]

    assert s.parent_storage_mode == "classic"
    assert s.parent_target_tokens == 900
    assert s.parent_overlap_tokens == 80
    assert s.parent_max_tokens == 1200
