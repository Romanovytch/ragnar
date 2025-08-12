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
    def __init__(self, repo_path: Path, base_url: str = "https://book.utilitr.org/"):
        self.repo_path = Path(repo_path)
        self.base_url = base_url.rstrip("/") + "/"
        self.commit = git_head_commit(self.repo_path)

    def iter_docs(self) -> Iterable[DocRecord]:
        patterns = ["**/*.md", "**/*.qmd"]
        ignore_dirs = {".git", "_book", "docs", ".quarto", "renv", ".github"}
        files: List[Path] = []
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
                m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
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
