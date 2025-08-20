from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Dict, Any, List
from ..util import read_text, extract_frontmatter, git_head_commit


@dataclass
class DocRecord:
    text: str
    metadata: Dict[str, Any]


class UtilitrSource:
    """Iterate over utilitR Markdown/Quarto documents with rich metadata.

    This adapter expects a local clone of the utilitR repository. It discovers
    `*.md` and `*.qmd` files (excluding build/system directories), splits away
    YAML frontmatter, infers a document title when missing, and constructs
    stable URLs for both the public book (`source_url`) and GitHub (`repo_url`)
    pinned to the current commit.

    Args:
        repo_path: Path to the local utilitR repository clone.
        base_url: Base URL of the published Quarto book (used to build
            `source_url` for each document). A trailing slash is ensured.

    Attributes:
        repo_path: Normalized repository path.
        base_url: Normalized base URL (with trailing slash).
        commit: Current git commit hash of the repository (or None if unknown).

    Note:
        The iterator yields `DocRecord(text, metadata)` for each discovered file.
        Downstream, the chunker will split `text` into atomic units while
        preserving `metadata` for citations.
    """
    def __init__(self, repo_path: Path, base_url: str = "https://book.utilitr.org/"):
        self.repo_path = Path(repo_path)
        self.base_url = base_url.rstrip("/") + "/"
        self.commit = git_head_commit(self.repo_path)

    def iter_docs(self) -> Iterable[DocRecord]:
        """Yield `DocRecord` items for each relevant Markdown/Quarto file.

        Discovery rules:

          - Include: all `**/*.md`, `**/*.qmd`.
          - Exclude directories: {".git", "_book", "docs", ".quarto", "renv", ".github"}.

        For each file:

          1. Read raw text and split YAML frontmatter (via `extract_frontmatter`).
          2. Determine title:
             - from frontmatter `title`, or
             - from the first Markdown `# Heading`, or
             - fallback to the filename stem.
          3. Compute URLs:
             - `source_url` → `{base_url}/{relative}.html`
             - `repo_url`   → GitHub blob URL pinned to `{commit}` (or "main").

        Yields:
            DocRecord: `text`=Markdown body (no frontmatter), `metadata` with keys
            described in the class docstring.

        Example:

            >>> src = UtilitrSource(Path("../utilitR"))
            >>> doc = next(iter(src.iter_docs()))
            >>> ("doc_title" in doc.metadata, doc.metadata["source"])
            (True, "utilitr")
        """

        patterns = ["**/*.md", "**/*.qmd"]
        ignore_dirs = {".git", "_book", "docs", ".quarto", "renv", ".github"}
        files: List[Path] = []

        # Add any file path matching patterns in `files` list
        for pat in patterns:
            files.extend(self.repo_path.glob(pat))

        for p in sorted(files):
            if any(part in ignore_dirs for part in p.parts):
                continue
            rel = p.relative_to(self.repo_path).as_posix()
            raw = read_text(p)
            fm, body = extract_frontmatter(raw)

            title = fm.get("title")
            if not title:
                m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)  # Find first heading
                title = m.group(1).strip() if m else p.stem

            html_rel = rel.rsplit(".", 1)[0] + ".html"
            source_url = self.base_url + html_rel
            repo_url = f"https://github.com/InseeFrLab/utilitR/blob/{self.commit or 'main'}/{rel}"

            meta = {
                "source": "utilitr",
                "source_type": "quarto_book",
                "file_path": rel,
                "doc_title": title,
                "source_url": source_url,
                "repo_url": repo_url,
                "git_commit": self.commit,
                "lang": fm.get("lang") or "fr",
            }
            yield DocRecord(text=body, metadata=meta)
