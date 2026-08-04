from __future__ import annotations

from collections.abc import Iterable
from typing import Literal
from urllib.parse import urlparse

import httpx
from openai import OpenAI

LLMProvider = Literal["openai", "mistral", "ollama", "other"]

_OLLAMA_PORTS = {11434, 11435}


def detect_llm_provider(
    api_base: str,
    fallback_provider_name: str | None = None,
) -> LLMProvider | str:
    """Infer the LLM provider from an OpenAI-compatible API base URL."""
    parsed = urlparse(api_base)
    hostname = (parsed.hostname or "").lower()

    if hostname == "api.openai.com" or hostname.endswith(".openai.com"):
        return "openai"
    if hostname == "api.mistral.ai" or hostname.endswith(".mistral.ai"):
        return "mistral"
    if fallback_provider_name and fallback_provider_name.strip():
        return fallback_provider_name.strip().lower()
    if hostname == "ollama" or hostname.startswith("ollama.") or parsed.port in _OLLAMA_PORTS:
        return "ollama"
    return "other"


def build_chat_provider_kwargs(
    api_base: str,
    reasoning_effort: str,
    fallback_provider_name: str | None = None,
) -> dict[str, object]:
    """Return Ollama-only extensions for an OpenAI-compatible chat request."""
    if (
        detect_llm_provider(api_base, fallback_provider_name) != "ollama"
        or not reasoning_effort
    ):
        return {}
    return {"extra_body": {"reasoning_effort": reasoning_effort}}


class ChatClient:
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str = "",
        thinking: str = "none",
        timeout: float = 120.0,
        insecure: bool = False,
        max_output_tokens: int | None = None,
        max_retries: int | None = None,
        empty_response_retries: int = 0,
        provider_name: str | None = None,
    ):
        if not api_base or not api_base.startswith(("http://", "https://")):
            raise ValueError("api_base must start with http(s)://")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if max_retries is not None and max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if empty_response_retries < 0:
            raise ValueError("empty_response_retries must be non-negative")

        client_kwargs: dict[str, object] = {
            "base_url": api_base.rstrip("/"),
            "api_key": api_key or "EMPTY",
            "timeout": timeout,
        }
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        if insecure:
            client_kwargs["http_client"] = httpx.Client(verify=False, timeout=timeout)

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.empty_response_retries = empty_response_retries
        self.provider = detect_llm_provider(api_base, provider_name)
        self.provider_kwargs = build_chat_provider_kwargs(api_base, thinking, provider_name)

    def complete_json(self, messages: list[dict[str, str]]) -> str:
        request_kwargs: dict[str, object] = {
            "temperature": 0,
            "response_format": {"type": "json_object"},
            **self.provider_kwargs,
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_tokens"] = self.max_output_tokens
        print(
            f"[debug] Sending LLM request with {len(messages)} messages "
            f"and kwargs: {request_kwargs}"
        )
        for attempt in range(self.empty_response_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **request_kwargs,
            )
            choices = response.choices
            if choices:
                content = choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content
            if attempt < self.empty_response_retries:
                print("[warn] LLM API returned an empty message; retrying")

        raise RuntimeError("LLM API returned an empty message")

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ) -> Iterable[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            **self.provider_kwargs,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
