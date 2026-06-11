from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from agora.sources.models.base import (
    DenseVectorConfig,
    MultiVectorConfig,
    SparseVectorConfig,
    VectorIndexConfig,
)

from .remote import RemoteOpenAIEncoder


@dataclass(frozen=True)
class SparseVectorData:
    indices: list[int]
    values: list[float]


DenseVectorBatch: TypeAlias = np.ndarray
SparseVectorBatch: TypeAlias = list[SparseVectorData]
MultiVectorBatch: TypeAlias = list[list[list[float]]]
NamedVectorBatch: TypeAlias = dict[str, DenseVectorBatch | SparseVectorBatch | MultiVectorBatch]


_FASTEMBED_LANGUAGE_ALIASES = {
    "ar": "arabic",
    "da": "danish",
    "de": "german",
    "du": "dutch",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "hu": "hungarian",
    "it": "italian",
    "nl": "dutch",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
}


def normalize_fastembed_language(language: str | None) -> str:
    if not language:
        return "english"
    normalized = language.strip().lower().replace("_", "-")
    normalized = normalized.split("-", 1)[0]
    return _FASTEMBED_LANGUAGE_ALIASES.get(normalized, normalized)


class FastEmbedSparseEncoder:
    """Sparse encoder backed by FastEmbed sparse text models."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        language: str = "english",
        disable_stemmer: bool = False,
    ) -> None:
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "Sparse vector mode requires FastEmbed. Install dependencies with "
                "`pip install -e .` or `pip install fastembed`."
            ) from e

        try:
            self._model = SparseTextEmbedding(
                model_name=model_name,
                language=language,
                disable_stemmer=disable_stemmer,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize FastEmbed sparse model '{model_name}'. "
                "The first run may need network access to download model files."
            ) from e

    def encode(self, texts: list[str], batch_size: int = 64) -> SparseVectorBatch:
        vectors: SparseVectorBatch = []
        for item in self._model.embed(texts, batch_size=batch_size):
            raw_indices = item.indices
            raw_values = item.values
            indices = raw_indices.tolist() if hasattr(raw_indices, "tolist") else raw_indices
            values = raw_values.tolist() if hasattr(raw_values, "tolist") else raw_values
            vectors.append(
                SparseVectorData(
                    indices=[int(i) for i in indices],
                    values=[float(v) for v in values],
                )
            )
        return vectors


class ConfiguredNamedVectorEncoder:
    """Build named vector outputs from the configured vector index."""

    def __init__(
        self,
        config: VectorIndexConfig,
        api_base: str,
        model: str,
        api_key: str = "",
        insecure: bool = False,
        default_lang: str | None = None,
    ) -> None:
        self.config = config
        self._dense_encoders: dict[str, RemoteOpenAIEncoder] = {}
        self._sparse_encoders: dict[str, FastEmbedSparseEncoder] = {}
        sparse_default_language = normalize_fastembed_language(default_lang)

        for vector in config.vectors:
            if isinstance(vector, DenseVectorConfig | MultiVectorConfig):
                self._dense_encoders[vector.name] = RemoteOpenAIEncoder(
                    api_base=api_base,
                    model=vector.model or model,
                    api_key=api_key,
                    insecure=insecure,
                )
            elif isinstance(vector, SparseVectorConfig):
                sparse_language = normalize_fastembed_language(
                    vector.language or sparse_default_language
                )
                self._sparse_encoders[vector.name] = FastEmbedSparseEncoder(
                    model_name=vector.model,
                    language=sparse_language,
                    disable_stemmer=vector.disable_stemmer,
                )

    def resolved_dimensions(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for vector in self.config.vectors:
            if isinstance(vector, SparseVectorConfig):
                continue
            configured_size = vector.size
            actual_size = self._dense_encoders[vector.name].dim
            if configured_size is not None and configured_size != actual_size:
                raise ValueError(
                    f"Configured vector '{vector.name}' has size {configured_size}, "
                    f"but provider returned {actual_size}"
                )
            dims[vector.name] = actual_size
        return dims

    def encode(self, texts: list[str], batch_size: int = 64) -> NamedVectorBatch:
        outputs: NamedVectorBatch = {}
        for vector in self.config.vectors:
            if isinstance(vector, DenseVectorConfig):
                outputs[vector.name] = self._dense_encoders[vector.name].encode(
                    texts, batch_size=batch_size
                )
            elif isinstance(vector, SparseVectorConfig):
                outputs[vector.name] = self._sparse_encoders[vector.name].encode(
                    texts, batch_size=batch_size
                )
            elif isinstance(vector, MultiVectorConfig):
                dense = self._dense_encoders[vector.name].encode(texts, batch_size=batch_size)
                outputs[vector.name] = [[row.tolist()] for row in dense]
        return outputs
