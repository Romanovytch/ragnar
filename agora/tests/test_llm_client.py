from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agora.llm.llm_client import (
    ChatClient,
    build_chat_provider_kwargs,
    detect_llm_provider,
)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://api.openai.com/v1", "openai"),
        ("http://localhost:11434/v1", "ollama"),
        ("http://localhost:11435/v1", "ollama"),
        ("http://ollama:11434/v1", "ollama"),
        ("https://api.mistral.ai/v1", "other"),
    ],
)
def test_detect_llm_provider(api_base, expected):
    assert detect_llm_provider(api_base) == expected


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.openai.com/v1",
        "https://api.mistral.ai/v1",
    ],
)
def test_provider_kwargs_omit_ollama_extension_for_other_providers(api_base):
    assert build_chat_provider_kwargs(api_base, "none") == {}


def test_ollama_json_request_retries_empty_response_and_streams():
    empty_response = SimpleNamespace(choices=[])
    json_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "One."}'))]
    )
    stream_response = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Streamed"))])
    ]

    with patch("agora.llm.llm_client.OpenAI") as openai_cls:
        create = openai_cls.return_value.chat.completions.create
        create.side_effect = [empty_response, json_response, stream_response]
        client = ChatClient(
            api_base="http://localhost:11435/v1",
            model="qwen3.5:9b",
            thinking="medium",
            max_output_tokens=512,
            max_retries=0,
            empty_response_retries=1,
        )

        assert client.complete_json([{"role": "user", "content": "Summarize"}])
        assert list(client.stream_chat([{"role": "user", "content": "Chat"}])) == ["Streamed"]

    openai_cls.assert_called_once_with(
        base_url="http://localhost:11435/v1",
        api_key="EMPTY",
        timeout=120.0,
        max_retries=0,
    )
    for call in create.call_args_list[:2]:
        assert call.kwargs["max_tokens"] == 512
        assert call.kwargs["extra_body"] == {"reasoning_effort": "medium"}
    assert create.call_args_list[2].kwargs["extra_body"] == {"reasoning_effort": "medium"}


def test_insecure_client_disables_tls_verification():
    with (
        patch("agora.llm.llm_client.httpx.Client") as http_client_cls,
        patch("agora.llm.llm_client.OpenAI") as openai_cls,
    ):
        ChatClient(
            api_base="http://localhost:11434/v1",
            model="qwen3.5:9b",
            insecure=True,
        )

    http_client_cls.assert_called_once_with(verify=False)
    assert openai_cls.call_args.kwargs["http_client"] is http_client_cls.return_value


def test_openai_request_omits_reasoning_effort():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary": "One."}'))]
    )

    with patch("agora.llm.llm_client.OpenAI") as openai_cls:
        create = openai_cls.return_value.chat.completions.create
        create.return_value = response
        client = ChatClient(
            api_base="https://api.openai.com/v1",
            model="gpt-4o",
            thinking="none",
        )

        client.complete_json([{"role": "user", "content": "Summarize"}])

    assert "extra_body" not in create.call_args.kwargs
    assert "reasoning_effort" not in create.call_args.kwargs
