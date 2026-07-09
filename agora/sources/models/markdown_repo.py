from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class MarkdownRepoConfig(BaseModel):
    """Config for a local Markdown/Quarto/Rmd repository (kind='markdown_repo').

    Required:
        kind: must be "markdown_repo"
        repo_path: local path to the cloned repo
        base_url: public site root (e.g., "https://book.utilitr.org/")

    Optional (fallback to shared defaults if unset):
        repo_url_template: template to build GitHub (or other) file URLs,
                           must contain'{path}'
        include_globs: file glob patterns to include
        exclude_dirs: directory names to ignore
        default_lang: default language code, e.g. "fr"
        html_path_template: map a source file path to its HTML page path.
            Default: "{path_no_ext}.html"
        frontmatter_title_keys: override the keys to look for title
        frontmatter_lang_keys: override the keys to look for language
    """

    kind: Literal["markdown_repo"]
    repo_path: Path
    base_url: str

    include_globs: list[str] | None = None
    exclude_dirs: list[str] | None = None
    default_lang: str | None = None

    repo_url_template: str | None = None
    html_path_template: str | None = "{path_no_ext}.html"
    frontmatter_title_keys: list[str] | None = None
    frontmatter_lang_keys: list[str] | None = None

    parent_storage_mode: Literal["none", "classic"] = "none"
    parent_target_tokens: int = 1000
    parent_overlap_tokens: int = 120
    parent_max_tokens: int = 1500

    synthetic_llm_summary: bool = False
    synthetic_llm_summary_group_max_tokens: int = 3000
    synthetic_llm_summary_max_sentences: int = 3

    @field_validator("repo_path")
    @classmethod
    def _path_exists(cls, p: Path) -> Path:
        if not p.exists() or not p.is_dir():
            raise ValueError(f"repo_path does not exist or is not a directory: {p}")
        return p

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        # normalize trailing slash
        return v.rstrip("/") + "/"

    @model_validator(mode="after")
    def _check_templates(self):
        t = self.repo_url_template or ""
        if t:
            if "{path}" not in t:
                raise ValueError("repo_url_template must contain {path} - {commit} is optional")
            if not (self.html_path_template and "{" in self.html_path_template):
                # Not strictly required, but warn early if it's a constant path
                # (we still allow it in case someone really wants a fixed page)
                pass
        if self.parent_target_tokens <= 0:
            raise ValueError("parent_target_tokens must be positive")
        if self.parent_overlap_tokens < 0:
            raise ValueError("parent_overlap_tokens must be non-negative")
        if self.parent_max_tokens < self.parent_target_tokens:
            raise ValueError(
                "parent_max_tokens must be greater than or equal to parent_target_tokens"
            )
        if self.synthetic_llm_summary_group_max_tokens <= 0:
            raise ValueError("synthetic_llm_summary_group_max_tokens must be positive")
        if self.synthetic_llm_summary_max_sentences <= 0:
            raise ValueError("synthetic_llm_summary_max_sentences must be positive")
        return self
