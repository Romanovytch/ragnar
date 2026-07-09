from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from agora.chunking import Chunk
from agora.util import count_tokens


@dataclass(frozen=True)
class SummaryChunkGroup:
    chunks: list[Chunk]
    text: str
    token_count: int


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    keywords: list[str]


class RemoteOpenAIChatClient:
    """Minimal client for OpenAI-compatible `/v1/chat/completions`."""

    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        insecure: bool = False,
    ) -> None:
        if not api_base or not api_base.startswith(("http://", "https://")):
            raise ValueError("api_base must start with http(s)://")
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.verify = not insecure
        self._session = requests.Session()

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = self._session.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify,
        )
        if response.status_code != 200:
            raise RuntimeError(f"LLM API {response.status_code}: {response.text[:500]}")
        choices = response.json().get("choices") or []
        if not choices:
            raise RuntimeError("LLM API returned no choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM API returned an empty message")
        return content


def build_summary_prompt(text: str, max_sentences: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You generate faithful retrieval summaries for technical documentation. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate a concise summary of the provided text.\n"
                "Preserve important concepts, entities, technical terms, and relevant "
                "expressions.\n"
                "Avoid vague or generic summaries.\n"
                f"Limit the summary to at most {max_sentences} sentences.\n"
                "Extract correlated keywords based on the context and meaning of the text, "
                "not only literal word extraction.\n"
                'Return exactly this JSON shape: {"summary": "...", "keywords": ["..."]}.\n\n'
                "Text:\n"
                f"{text}"
            ),
        },
    ]


def group_chunks_for_summary(chunks: list[Chunk], max_tokens: int) -> list[SummaryChunkGroup]:
    groups: list[SummaryChunkGroup] = []
    current: list[Chunk] = []
    current_tokens = 0
    current_doc_key: tuple[Any, ...] | None = None

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk.text)
        doc_key = (
            chunk.metadata.get("source"),
            chunk.metadata.get("file_path"),
            chunk.metadata.get("git_commit"),
        )
        should_flush = bool(current) and (
            doc_key != current_doc_key or current_tokens + chunk_tokens > max_tokens
        )
        if should_flush:
            groups.append(_make_group(current, current_tokens))
            current = []
            current_tokens = 0

        current.append(chunk)
        current_tokens += chunk_tokens
        current_doc_key = doc_key

    if current:
        groups.append(_make_group(current, current_tokens))
    return groups


def generate_summary(client: RemoteOpenAIChatClient, group: SummaryChunkGroup, max_sentences: int):
    prompt = build_summary_prompt(group.text, max_sentences=max_sentences)
    return parse_summary_response(client.complete_json(prompt), max_sentences=max_sentences)


def parse_summary_response(content: str, max_sentences: int) -> SummaryResult:
    raw = _strip_json_fence(content.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("LLM summary response must be valid JSON") from e

    summary = data.get("summary")
    keywords = data.get("keywords")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("LLM summary response must include a non-empty 'summary'")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValueError("LLM summary response must include string list 'keywords'")

    return SummaryResult(
        summary=_limit_sentences(summary.strip(), max_sentences),
        keywords=[k.strip() for k in keywords if k.strip()],
    )


def build_summary_chunks(
    groups: list[SummaryChunkGroup],
    results: list[SummaryResult],
) -> list[Chunk]:
    if len(groups) != len(results):
        raise ValueError("summary groups and results must have the same length")

    summary_chunks: list[Chunk] = []
    for idx, (group, result) in enumerate(zip(groups, results, strict=True)):
        first = group.chunks[0]
        source_chunk_ids = [chunk.id for chunk in group.chunks]
        source_chunk_indexes = [
            chunk.metadata.get("chunk_index")
            for chunk in group.chunks
            if "chunk_index" in chunk.metadata
        ]
        summary_id = _make_summary_id(source_chunk_ids)
        metadata = first.metadata | {
            "summary": result.summary,
            "keywords": result.keywords,
            "source_chunk_ids": source_chunk_ids,
            "source_chunk_indexes": source_chunk_indexes,
            "summary_index": idx,
            "token_count": count_tokens(result.summary),
        }
        summary_chunks.append(Chunk(id=summary_id, text=result.summary, metadata=metadata))
    return summary_chunks


def _make_group(chunks: list[Chunk], token_count: int) -> SummaryChunkGroup:
    text = "\n\n".join(f"[chunk_id={chunk.id}]\n{chunk.text}" for chunk in chunks)
    return SummaryChunkGroup(chunks=list(chunks), text=text, token_count=token_count)


def _make_summary_id(source_chunk_ids: list[str]) -> str:
    ns = uuid.uuid5(uuid.NAMESPACE_URL, "https://book.utilitr.org")
    name = "summary:" + "|".join(source_chunk_ids)
    return str(uuid.uuid5(ns, name))


def _strip_json_fence(content: str) -> str:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, count=1)
        content = re.sub(r"\s*```$", "", content, count=1)
    return content


def _limit_sentences(summary: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        raise ValueError("max_sentences must be positive")
    sentences = re.findall(r"[^.!?]+[.!?]?(?:\s+|$)", summary)
    if len(sentences) <= max_sentences:
        return summary
    limited = "".join(sentences[:max_sentences]).strip()
    return limited or summary
