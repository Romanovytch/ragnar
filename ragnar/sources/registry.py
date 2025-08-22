from __future__ import annotations
from typing import Callable, Dict, Any
from ragnar.sources.models.markdown_repo import MarkdownRepoConfig
from ragnar.sources.markdown_source import MarkdownRepoSource

# name + typed cfg -> Source instance
SourceFactory = Callable[[str, Any], Any]
_REGISTRY: Dict[str, SourceFactory] = {}


def register(kind: str, factory: SourceFactory) -> None:
    _REGISTRY[kind] = factory


def get_registry() -> Dict[str, SourceFactory]:
    return dict(_REGISTRY)


def build_source(name: str, cfg: Any):
    kind = getattr(cfg, "kind", None)
    if kind not in _REGISTRY:
        raise ValueError(f"Unsupported source kind '{kind}' for source '{name}'")
    return _REGISTRY[kind](name, cfg)


# --- built-in kinds ---

def _mk_markdown_repo(name: str, cfg: MarkdownRepoConfig):
    return MarkdownRepoSource(
        name=name,
        repo_path=cfg.repo_path,
        include_globs=cfg.include_globs or ["**/*.md", "**/*.qmd", "**/*.Rmd"],
        exclude_dirs=cfg.exclude_dirs or [],
        base_url=cfg.base_url,
        html_path_template=cfg.html_path_template or "{path_no_ext}.html",
        repo_url_template=cfg.repo_url_template,
        default_lang=cfg.default_lang or "en",
        frontmatter_title_keys=cfg.frontmatter_title_keys,
        frontmatter_lang_keys=cfg.frontmatter_lang_keys,
        follow_symlinks=False,
    )


register("markdown_repo", _mk_markdown_repo)
