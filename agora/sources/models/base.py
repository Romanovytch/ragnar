from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class DenseVectorConfig(BaseModel):
    name: str = "dense"
    kind: Literal["dense"] = "dense"
    size: int | None = None
    distance: Literal["cosine"] = "cosine"
    model: str | None = None


class SparseVectorConfig(BaseModel):
    name: str = "sparse"
    kind: Literal["sparse"] = "sparse"
    provider: Literal["fastembed"] = "fastembed"
    model: str = "Qdrant/bm25"
    modifier: Literal["idf", "none"] = "idf"
    language: str | None = None
    disable_stemmer: bool = False


class MultiVectorConfig(BaseModel):
    name: str = "multi"
    kind: Literal["multi"] = "multi"
    provider: Literal["fastembed"] = "fastembed"
    model: str = "answerdotai/answerai-colbert-small-v1"
    size: int | None = None
    distance: Literal["cosine"] = "cosine"
    comparator: Literal["max_sim"] = "max_sim"


VectorModeConfig = Annotated[
    DenseVectorConfig | SparseVectorConfig | MultiVectorConfig,
    Field(discriminator="kind"),
]


class VectorIndexConfig(BaseModel):
    vectors: list[VectorModeConfig] = Field(
        default_factory=lambda: [DenseVectorConfig()]
    )

    @model_validator(mode="after")
    def _check_unique_names(self):
        names = [v.name for v in self.vectors]
        if not names:
            raise ValueError("vector_index must configure at least one vector")
        if len(names) != len(set(names)):
            raise ValueError("vector_index vector names must be unique")
        return self


class SourceDefaults(BaseModel):
    """Cross-kind defaults (safe for any file-based source)."""

    include_globs: list[str] = ["**/*.md", "**/*.qmd", "**/*.Rmd"]
    exclude_dirs: list[str] = [
        ".git",
        "_book",
        "docs",
        ".quarto",
        "renv",
        ".github",
        "node_modules",
        "build",
        "dist",
        "site",
    ]
    default_lang: str = "en"
    follow_symlinks: bool = False


class SourcesConfig(BaseModel):
    """Top-level configuration with shared defaults and named sources.

    `sources` is a mapping: source name -> typed config (discriminated by `kind`).
    The discriminated union is assembled in the loader, so we keep the field
    shape here and let the loader fill the type info.
    """

    version: int = 1
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    defaults: SourceDefaults = Field(default_factory=SourceDefaults)
    # The loader will parse this into a discriminated union.
    sources: dict[str, dict]
