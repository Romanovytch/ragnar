from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional
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

    include_globs: Optional[List[str]] = None
    exclude_dirs: Optional[List[str]] = None
    default_lang: Optional[str] = None

    repo_url_template: Optional[str] = None
    html_path_template: Optional[str] = "{path_no_ext}.html"
    frontmatter_title_keys: Optional[List[str]] = None
    frontmatter_lang_keys: Optional[List[str]] = None

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
        return self
