from __future__ import annotations

import sys
import types

import numpy as np

from agora.embeddings.named import (
    ConfiguredNamedVectorEncoder,
    FastEmbedSparseEncoder,
    normalize_fastembed_language,
)
from agora.sources.models.base import SparseVectorConfig, VectorIndexConfig


class _FakeSparseEmbedding:
    def __init__(self, indices, values):
        self.indices = np.array(indices, dtype=np.int32)
        self.values = np.array(values, dtype=np.float32)


class _FakeSparseTextEmbedding:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def embed(self, texts, batch_size=64):
        assert list(texts) == ["alpha beta", "gamma"]
        assert batch_size == 7
        yield _FakeSparseEmbedding([3, 1], [0.25, 1.5])
        yield _FakeSparseEmbedding([4], [2.0])


def test_fastembed_sparse_encoder_converts_embeddings(monkeypatch):
    fake_fastembed = types.SimpleNamespace(SparseTextEmbedding=_FakeSparseTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    encoder = FastEmbedSparseEncoder(
        model_name="Qdrant/bm25",
        language="french",
        disable_stemmer=True,
    )
    vectors = encoder.encode(["alpha beta", "gamma"], batch_size=7)

    assert _FakeSparseTextEmbedding.init_kwargs == {
        "model_name": "Qdrant/bm25",
        "language": "french",
        "disable_stemmer": True,
    }
    assert vectors[0].indices == [3, 1]
    assert vectors[0].values == [0.25, 1.5]
    assert vectors[1].indices == [4]
    assert vectors[1].values == [2.0]


def test_configured_encoder_uses_source_default_lang_for_sparse(monkeypatch):
    fake_fastembed = types.SimpleNamespace(SparseTextEmbedding=_FakeSparseTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    ConfiguredNamedVectorEncoder(
        config=VectorIndexConfig(vectors=[SparseVectorConfig(name="sparse")]),
        api_base="http://emb/v1",
        model="dense-model",
        default_lang="fr",
    )

    assert _FakeSparseTextEmbedding.init_kwargs["language"] == "french"


def test_sparse_vector_language_overrides_source_default_lang(monkeypatch):
    fake_fastembed = types.SimpleNamespace(SparseTextEmbedding=_FakeSparseTextEmbedding)
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    ConfiguredNamedVectorEncoder(
        config=VectorIndexConfig(
            vectors=[SparseVectorConfig(name="sparse", language="spanish")]
        ),
        api_base="http://emb/v1",
        model="dense-model",
        default_lang="fr",
    )

    assert _FakeSparseTextEmbedding.init_kwargs["language"] == "spanish"


def test_normalize_fastembed_language_handles_source_lang_codes():
    assert normalize_fastembed_language("fr") == "french"
    assert normalize_fastembed_language("fr-FR") == "french"
    assert normalize_fastembed_language("en") == "english"
    assert normalize_fastembed_language("french") == "french"
