from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agora.chunking import Chunk
from agora.cli.ingest import _summary_collection_name
from agora.embeddings.named import SparseVectorData
from agora.sources.models.base import DenseVectorConfig, SparseVectorConfig, VectorIndexConfig
from agora.summary import (
    RemoteOpenAIChatClient,
    SummaryResult,
    build_summary_chunks,
    build_summary_embedding_text,
    build_summary_prompt,
    generate_summary,
    group_chunks_for_summary,
    parse_summary_response,
)
from agora.vectorstores.qdrant_store import build_named_points, upsert_named


def _chunk(idx: int, text: str = "alpha beta gamma", path: str = "doc.md") -> Chunk:
    return Chunk(
        id=f"11111111-1111-5111-8111-11111111111{idx}",
        text=text,
        metadata={
            "source": "docs",
            "source_type": "markdown_repo",
            "file_path": path,
            "doc_title": "Doc",
            "source_url": "https://example.org/doc.html",
            "repo_url": "https://git/doc.md",
            "git_commit": "abc",
            "lang": "en",
            "chapter": "Doc",
            "section": "Section",
            "breadcrumbs": ["Doc", "Section"],
            "token_count": 3,
            "url": "https://example.org/doc.html#section",
            "chunk_index": idx,
        },
    )


def test_group_chunks_for_summary_keeps_source_chunk_references_and_doc_boundaries():
    chunks = [
        _chunk(0, "alpha " * 10, path="a.md"),
        _chunk(1, "beta " * 10, path="a.md"),
        _chunk(2, "gamma " * 10, path="b.md"),
    ]

    groups = group_chunks_for_summary(chunks, max_tokens=30)

    assert [[chunk.id for chunk in group.chunks] for group in groups] == [
        [chunks[0].id, chunks[1].id],
        [chunks[2].id],
    ]
    assert f"[chunk_id={chunks[0].id}]" in groups[0].text
    assert f"[chunk_id={chunks[1].id}]" in groups[0].text


def test_summary_prompt_requires_json_summary_keywords_and_sentence_limit():
    messages = build_summary_prompt("Important text", max_sentences=2)
    prompt = "\n".join(message["content"] for message in messages)

    assert "summary" in prompt
    assert "keywords" in prompt
    assert "at most 2 sentences" in prompt
    assert "correlated keywords" in prompt
    assert "Return only valid JSON" in prompt
    assert "same language as the source text" in prompt


def test_remote_chat_client_uses_openai_sdk_with_custom_base_url():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "One."}'))]
    )

    with (
        patch("agora.summary.DefaultHttpxClient") as http_client_cls,
        patch("agora.summary.OpenAI") as openai_cls,
    ):
        openai_cls.return_value.chat.completions.create.return_value = response
        client = RemoteOpenAIChatClient(
            api_base="http://127.0.0.1:11434/v1",
            model="qwen3.5:9b",
            insecure=True,
        )

        content = client.complete_json([{"role": "user", "content": "Summarize"}])

    http_client_cls.assert_called_once_with(verify=False)
    openai_cls.assert_called_once_with(
        base_url="http://127.0.0.1:11434/v1/",
        api_key="not-needed",
        timeout=60.0,
        max_retries=0,
        http_client=http_client_cls.return_value,
    )
    openai_cls.return_value.chat.completions.create.assert_called_once_with(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "Summarize"}],
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
        reasoning_effort="none",
    )
    assert content == '{"summary": "One."}'


def test_remote_chat_client_retries_an_empty_response_once():
    empty_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
    valid_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "Retried."}'))]
    )

    with (
        patch("agora.summary.DefaultHttpxClient"),
        patch("agora.summary.OpenAI") as openai_cls,
    ):
        create = openai_cls.return_value.chat.completions.create
        create.side_effect = [empty_response, valid_response]
        client = RemoteOpenAIChatClient(
            api_base="http://127.0.0.1:11434/v1",
            model="qwen3.5:9b",
        )

        content = client.complete_json([{"role": "user", "content": "Summarize"}])

    assert create.call_count == 2
    assert content == '{"summary": "Retried."}'


def test_parse_summary_response_caps_sentences_from_config():
    result = parse_summary_response(
        '{"summary": "One. Two. Three.", "keywords": ["alpha", " beta "]}',
        max_sentences=2,
    )

    assert result.summary == "One. Two."
    assert result.keywords == ["alpha", "beta"]


def test_parse_summary_response_extracts_json_from_extra_text():
    result = parse_summary_response(
        'Here is the JSON: {"summary": "One.", "keywords": ["alpha"]} Thanks.',
        max_sentences=2,
    )

    assert result.summary == "One."
    assert result.keywords == ["alpha"]


def test_generate_summary_repairs_invalid_json_response():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def complete_json(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return '{"summary": "One", "keywords": ["alpha" "beta"]}'
            return '{"summary": "One.", "keywords": ["alpha", "beta"]}'

    client = FakeClient()
    group = group_chunks_for_summary([_chunk(0)], max_tokens=100)[0]

    result = generate_summary(client, group, max_sentences=1)

    assert result.summary == "One."
    assert result.keywords == ["alpha", "beta"]
    assert len(client.calls) == 2
    assert "repair" in client.calls[1][0]["content"].lower()


def test_build_summary_chunks_separates_retrieval_and_ordered_source_content():
    chunks = [_chunk(0, "First original child."), _chunk(1, "Second original child.")]
    groups = group_chunks_for_summary(chunks, max_tokens=100)
    summary_chunks = build_summary_chunks(
        groups,
        [SummaryResult(summary="Concise technical summary.", keywords=["alpha", "concept"])],
    )

    summary = summary_chunks[0]

    assert summary.text == "First original child.\n\n---\n\nSecond original child."
    assert build_summary_embedding_text(summary) == ("Concise technical summary.\n\nalpha, concept")
    assert summary.metadata["source"] == chunks[0].metadata["source"]
    assert summary.metadata["source_type"] == chunks[0].metadata["source_type"]
    assert summary.metadata["file_path"] == chunks[0].metadata["file_path"]
    assert summary.metadata["summary"] == "Concise technical summary."
    assert summary.metadata["keywords"] == ["alpha", "concept"]
    assert summary.metadata["source_chunk_ids"] == [chunks[0].id, chunks[1].id]
    assert summary.metadata["source_chunk_indexes"] == [0, 1]


def test_summary_points_use_separate_collection_and_named_dense_sparse_vectors():
    class FakeClient:
        kwargs = None

        def upload_points(self, **kwargs):
            self.kwargs = kwargs

    chunks = [_chunk(0)]
    groups = group_chunks_for_summary(chunks, max_tokens=100)
    summary_chunks = build_summary_chunks(
        groups,
        [SummaryResult(summary="Summary text.", keywords=["summary"])],
    )
    config = VectorIndexConfig(
        vectors=[DenseVectorConfig(name="dense", size=2), SparseVectorConfig(name="sparse")]
    )
    vectors = {
        "dense": np.array([[1.0, 2.0]], dtype="float32"),
        "sparse": [SparseVectorData(indices=[1], values=[0.5])],
    }

    points = build_named_points(summary_chunks, config, vectors, {"dense": 2})

    assert set(points[0].vector) == {"dense", "sparse"}
    assert points[0].payload["text"] == chunks[0].text
    assert points[0].payload["summary"] == "Summary text."
    assert points[0].payload["keywords"] == ["summary"]
    assert points[0].payload["source_chunk_ids"] == [chunks[0].id]

    client = FakeClient()
    upsert_named(
        client,
        _summary_collection_name("kb"),
        summary_chunks,
        config,
        vectors,
        {"dense": 2},
    )

    assert client.kwargs["collection_name"] == "kb_summaries"
    assert len(client.kwargs["points"]) == 1
