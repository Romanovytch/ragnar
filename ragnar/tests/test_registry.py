from __future__ import annotations
from pathlib import Path
from ragnar.sources.models.markdown_repo import MarkdownRepoConfig
from ragnar.sources.registry import build_source


def test_registry_builds_markdown_repo(tmp_path: Path):
    repo = tmp_path / "r"
    repo.mkdir()
    cfg = MarkdownRepoConfig(
        kind="markdown_repo",
        repo_path=repo,
        base_url="https://docs.example.org/",
        repo_url_template=None,
        include_globs=["**/*.md"],
        default_lang="fr",
    )
    src = build_source("s", cfg)
    # The adapter exposes iter_docs(); minimal assertion
    assert hasattr(src, "iter_docs")
