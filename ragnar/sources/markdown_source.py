from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import re

from ragnar.util import read_text, extract_frontmatter, git_head_commit


@dataclass
class DocRecord:
    """Single document unit produced by a Source adapter.

    Attributes:
        text: Plain markdown body (frontmatter removed).
        metadata: Provenance & citation fields suitable for payload storage.
    """
    text: str
    metadata: Dict[str, object]


class MarkdownRepoSource:
    """Generic adapter for local Markdown/Quarto/Rmd repositories.

    This class discovers files under a local `repo_path` using the provided
    include/exclude rules, extracts frontmatter, infers a title, and emits
    a uniform :class:`DocRecord` with provenance metadata.

    The resulting metadata matches the keys used by the legacy utilitR adapter
    (shape-compatible) with `source_type="markdown_repo"`.

    Parameters
    ----------
    name :
        Logical source name (used in `metadata["source"]`).
    repo_path :
        Local filesystem path to the cloned documentation repository.
        Can be absolute or relative (resolve before passing if needed).
    include_globs :
        Glob patterns (relative to `repo_path`) to include, e.g. `["**/*.md"]`.
    exclude_dirs :
        Directory *names* to ignore anywhere in the tree (e.g., `".git"`).
    base_url :
        Public site root (e.g., `"https://docs.example.org/"`). Trailing slash
        is normalized if missing.
    html_path_template :
        Template that maps a source file path to its HTML path under `base_url`.
        Placeholders available: `{path}`, `{path_no_ext}`, `{dir}`, `{stem}`.
        Defaults to `"{path_no_ext}.html"`.
    repo_url_template :
        Optional template to link back to the source repository. If provided,
        it should at least contain `{path}`; `{commit}` is supported but optional.
        If omitted, `repo_url` will be `None`.
    default_lang :
        Default language code used if frontmatter has no language.
    frontmatter_title_keys :
        Keys to look for the page title in frontmatter. Defaults `["title","pagetitle"]`.
    frontmatter_lang_keys :
        Keys to look for the language in frontmatter. Defaults `["lang","language"]`.
    default_branch :
        Branch name used when formatting `repo_url_template` if no git commit
        could be read from the repo. Defaults to `"main"`.
    emit_repo_url :
        Whether to emit `repo_url` in metadata when a template is provided.
        Defaults to `True`.
    follow_symlinks :
        Best-effort guard: when `False`, skip files that are symlinks or whose
        *immediate* parent directory is a symlink. (Recursive glob on symlinked
        directories is not followed by most setups; this is an extra precaution.)

    Notes
    -----
    **Metadata contract** (keys produced per document):

    - `source`: the `name` provided to the constructor.
    - `source_type`: `"markdown_repo"`.
    - `file_path`: POSIX-style relative path under `repo_path`.
    - `doc_title`: inferred title (frontmatter → first H1 → filename stem).
    - `source_url`: `base_url` + `html_path_template` mapping.
    - `repo_url`: formatted from `repo_url_template` (or `None`).
    - `git_commit`: commit hash (or `None`).
    - `lang`: language (frontmatter → `default_lang`).
    """

    def __init__(
        self,
        name: str,
        repo_path: Path,
        include_globs: List[str],
        exclude_dirs: List[str],
        base_url: str,
        html_path_template: str = "{path_no_ext}.html",
        repo_url_template: Optional[str] = None,
        default_lang: str = "en",
        frontmatter_title_keys: Optional[List[str]] = None,
        frontmatter_lang_keys: Optional[List[str]] = None,
        default_branch: str = "main",
        emit_repo_url: bool = True,
        follow_symlinks: bool = False,
    ) -> None:
        self.name = name
        self.repo_path = Path(repo_path)
        self.include_globs = include_globs or ["**/*.md", "**/*.qmd", "**/*.Rmd"]
        self.exclude_dirs = set(exclude_dirs or [])
        self.base_url = (base_url.rstrip("/") + "/") if base_url else ""
        self.html_path_template = html_path_template or "{path_no_ext}.html"
        self.repo_url_template = repo_url_template
        self.default_lang = default_lang or "en"
        self.frontmatter_title_keys = frontmatter_title_keys or ["title", "pagetitle"]
        self.frontmatter_lang_keys = frontmatter_lang_keys or ["lang", "language"]
        self.default_branch = default_branch or "main"
        self.emit_repo_url = emit_repo_url
        self.follow_symlinks = follow_symlinks

        self.commit = git_head_commit(self.repo_path)

    def _iter_files(self) -> List[Path]:
        files: List[Path] = []
        for pat in self.include_globs:
            files.extend(self.repo_path.glob(pat))
        out: List[Path] = []
        for p in sorted(set(files)):
            if not p.is_file():
                continue
            # symlink guard (best effort)
            if not self.follow_symlinks:
                if p.is_symlink():
                    continue
                parent = p.parent
                try:
                    if parent.is_symlink():
                        continue
                except Exception:
                    pass
            # directory name exclusion (match any path part)
            if any(part in self.exclude_dirs for part in p.parts):
                continue
            out.append(p)
        return out

    def _infer_title(self, fm: Dict[str, object], body: str, stem: str) -> str:
        # 1) frontmatter keys
        for k in self.frontmatter_title_keys:
            v = fm.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # 2) first ATX H1
        m = re.search(r"(?m)^\s*#\s+(.+?)\s*$", body)
        if m:
            return m.group(1).strip()
        # 3) fallback to filename stem
        return stem

    def _infer_lang(self, fm: Dict[str, object], default: str) -> str:
        for k in self.frontmatter_lang_keys:
            v = fm.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return default

    def _build_source_url(self, rel_posix: str) -> str:
        path_no_ext = rel_posix.rsplit(".", 1)[0]
        stem = Path(rel_posix).stem
        d = str(Path(rel_posix).parent).replace("\\", "/")
        mapping = {
            "path": rel_posix,
            "path_no_ext": path_no_ext,
            "dir": "" if d == "." else d,
            "stem": stem,
        }
        return self.base_url + self.html_path_template.format(**mapping).lstrip("/")

    def _build_repo_url(
        self, rel_posix: str, branch: str, commit: Optional[str]
    ) -> Optional[str]:
        if not (self.emit_repo_url and self.repo_url_template):
            return None
        path_no_ext = rel_posix.rsplit(".", 1)[0]
        stem = Path(rel_posix).stem
        d = str(Path(rel_posix).parent).replace("\\", "/")
        mapping = {
            "path": rel_posix,
            "path_no_ext": path_no_ext,
            "dir": "" if d == "." else d,
            "stem": stem,
            "branch": branch,
            "commit": commit or branch,
        }
        try:
            return self.repo_url_template.format(**mapping)
        except Exception:
            return None

    def iter_docs(self) -> Iterable[DocRecord]:
        """Yield :class:`DocRecord` for each discovered markdown-like file.

        Yields
        ------
        DocRecord
            Frontmatter-stripped body and metadata ready for indexing.

        Examples
        --------
        >>> src = MarkdownRepoSource(
        ...     name="utilitr",
        ...     repo_path=Path("../utilitR"),
        ...     include_globs=["**/*.md", "**/*.qmd"],
        ...     exclude_dirs=[".git", "_book", "docs", ".quarto", "renv", ".github"],
        ...     base_url="https://book.utilitr.org/",
        ...     repo_url_template="https://github.com/InseeFrLab/utilitR/blob/{commit}/{path}",
        ...     default_lang="fr",
        ... )
        >>> rec = next(iter(src.iter_docs()))
        >>> sorted(rec.metadata.keys())[:4]
        ['doc_title', 'file_path', 'git_commit', 'lang']
        """
        branch = self.default_branch
        for p in self._iter_files():
            rel_posix = p.relative_to(self.repo_path).as_posix()
            raw = read_text(p)
            fm, body = extract_frontmatter(raw)

            title = self._infer_title(fm, body, Path(rel_posix).stem)
            lang = self._infer_lang(fm, self.default_lang)

            source_url = self._build_source_url(rel_posix)
            repo_url = self._build_repo_url(rel_posix, branch, self.commit)

            meta = {
                "source": self.name,
                "source_type": "markdown_repo",
                "file_path": rel_posix,
                "doc_title": title,
                "source_url": source_url,
                "repo_url": repo_url,
                "git_commit": self.commit,
                "lang": lang,
            }
            yield DocRecord(text=body, metadata=meta)
