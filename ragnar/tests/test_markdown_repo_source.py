from __future__ import annotations
from pathlib import Path
import textwrap

from ragnar.sources.markdown_source import MarkdownRepoSource, DocRecord


def _w(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s).lstrip(), encoding="utf-8")


def test_markdown_repo_basic(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / ".git").mkdir(parents=True, exist_ok=True)
    _w(
        repo / "chapters" / "intro.md",
        """
        ---
        title: Bien démarrer
        lang: fr
        ---
        # Heading will be ignored because frontmatter title exists

        Corps **markdown**.
        """,
    )
    _w(
        repo / "guide.qmd",
        """
        # Pas de frontmatter ici
        Du texte simple.
        """,
    )

    src = MarkdownRepoSource(
        name="utilitr",
        repo_path=repo,
        include_globs=["**/*.md", "**/*.qmd"],
        exclude_dirs=[".git", "_book", "docs", ".quarto", "renv", ".github"],
        base_url="https://book.utilitr.org/",
        repo_url_template="https://github.com/InseeFrLab/utilitR/blob/{commit}/{path}",
        default_lang="fr",
    )

    docs = list(src.iter_docs())
    assert len(docs) == 2
    for d in docs:
        assert isinstance(d, DocRecord)
        assert "text" in d.__dict__
        m = d.metadata
        # shape compatibility with utilitR adapter
        for k in ["source", "source_type", "file_path", "doc_title", "source_url",
                  "repo_url", "git_commit", "lang"]:
            assert k in m

    # Check title inference & URLs
    d0 = next(x for x in docs if x.metadata["file_path"] == "chapters/intro.md")
    assert d0.metadata["doc_title"] == "Bien démarrer"
    assert d0.metadata["source_url"].endswith("chapters/intro.html")

    d1 = next(x for x in docs if x.metadata["file_path"] == "guide.qmd")
    # title via first H1
    assert d1.metadata["doc_title"] == "Pas de frontmatter ici"
    assert d1.metadata["source_url"].endswith("guide.html")


def test_markdown_repo_respects_exclude_dirs_and_symlinks(tmp_path: Path):
    repo = tmp_path / "repo"
    secret = repo / "docs"
    secret.mkdir(parents=True)
    _w(secret / "hide.md", "# Hidden\n")

    _w(repo / "ok.md", "# OK\n")

    # Symlinked file should be skipped when follow_symlinks=False
    (repo / "link.md").symlink_to(repo / "ok.md")

    src = MarkdownRepoSource(
        name="s",
        repo_path=repo,
        include_globs=["**/*.md"],
        exclude_dirs=["docs"],  # exclude "docs" dir name anywhere
        base_url="https://x.example/",
    )

    files = [d.metadata["file_path"] for d in src.iter_docs()]
    assert files == ["ok.md"]  # hide.md excluded; link.md skipped
