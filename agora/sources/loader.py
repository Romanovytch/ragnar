from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, Field

from .models.base import SourceDefaults, VectorIndexConfig
from .models.base import SourcesConfig as _RawSourcesConfig
from .models.markdown_repo import MarkdownRepoConfig

SourceConfigUnion = Annotated[
    MarkdownRepoConfig,  # extend here with future kinds
    Field(discriminator="kind"),
]


class SourcesConfig(BaseModel):
    version: int = 1
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    defaults: SourceDefaults
    sources: dict[str, SourceConfigUnion]


def load_sources_config(path: Path) -> SourcesConfig:
    """Load YAML and return a validated SourcesConfig (typed, discriminated by `kind`)."""
    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot read sources config at: {path}") from e

    data = yaml.safe_load(raw_text) or {}
    raw = _RawSourcesConfig.model_validate(data)

    # --- PRE-NORMALIZE relative paths against the YAML file location ---
    base_dir = path.parent
    norm_sources: dict[str, dict] = {}
    for name, cfg in (raw.sources or {}).items():
        cfg = dict(cfg)  # shallow copy
        if cfg.get("kind") == "markdown_repo":
            rp = cfg.get("repo_path")
            if isinstance(rp, str):
                rp_path = Path(rp)
                if not rp_path.is_absolute():
                    cfg["repo_path"] = str((base_dir / rp_path).resolve())
        norm_sources[name] = cfg

    typed = SourcesConfig.model_validate(
        {
            "version": raw.version,
            "vector_index": raw.vector_index.model_dump(),
            "defaults": raw.defaults.model_dump(),
            "sources": norm_sources,
        }
    )
    return typed


def _merge_defaults(cfg: BaseModel, defaults: BaseModel) -> BaseModel:
    """Generic overlay:
    For every key in `defaults`, if the key exists as a field on `cfg`
    and is missing/None in cfg, copy the default value in.
    Unknown keys are ignored.
    """
    cfg_fields = set(type(cfg).model_fields.keys())
    current = cfg.model_dump(exclude_unset=True)
    update: dict[str, Any] = {}

    for k, v in defaults.model_dump().items():
        if k in cfg_fields and (k not in current or current[k] is None):
            update[k] = v

    return cfg.model_copy(update=update)


def resolve_source(
    name: str, cfg: SourceConfigUnion, defaults: SourceDefaults
) -> SourceConfigUnion:
    # Apply generic default merge (works for any BaseModel kind)
    cfg = _merge_defaults(cfg, defaults)

    # Kind-specific post-processing (optional)
    if isinstance(cfg, MarkdownRepoConfig):
        return cfg

    raise ValueError(f"Unsupported source kind for '{name}': {getattr(cfg, 'kind', '?')}")


def validate_and_resolve(path: Path) -> dict[str, SourceConfigUnion]:
    sc = load_sources_config(path)
    return resolve_sources_config(sc)


def resolve_sources_config(sc: SourcesConfig) -> dict[str, SourceConfigUnion]:
    resolved: dict[str, SourceConfigUnion] = {}
    for name, cfg in sc.sources.items():
        cfg = resolve_source(name, cfg, sc.defaults)
        resolved[name] = cfg
    return resolved
