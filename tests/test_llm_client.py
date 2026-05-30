from __future__ import annotations

import json

import pytest

from ashare_cross_section_similarity.llm_client import (
    DEFAULT_LLM_PROVIDER,
    LLMAPIError,
    LLMClient,
    LLMConfig,
    LLMMissingAPIKeyError,
    provider_presets,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_provider_presets_cover_requested_vendors() -> None:
    presets = provider_presets()

    assert DEFAULT_LLM_PROVIDER == "deepseek"
    assert set(presets) == {"deepseek", "qwen", "openai", "claude"}
    assert presets["qwen"].default_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert presets["openai"].api_key_env_var == "OPENAI_API_KEY"
    assert presets["claude"].api_style == "anthropic_messages"


def test_llm_client_builds_openai_compatible_payload_for_qwen() -> None:
    calls: list[dict[str, object]] = []

    def fake_transport(request, timeout: float):
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeResponse({"choices": [{"message": {"content": "{\"review\":\"ok\"}"}}]})

    client = LLMClient(
        LLMConfig(provider="qwen", api_key="sk-test", model="qwen-plus", thinking=True),
        transport=fake_transport,
    )

    content = client.chat([{"role": "user", "content": "hello"}])

    assert content == "{\"review\":\"ok\"}"
    assert calls[0]["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert calls[0]["timeout"] == 60.0
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["body"]["model"] == "qwen-plus"
    assert calls[0]["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}
    assert "thinking" not in calls[0]["body"]


def test_llm_client_builds_deepseek_payload_with_thinking() -> None:
    calls: list[dict[str, object]] = []

    def fake_transport(request, _timeout: float):
        calls.append({"url": request.full_url, "body": json.loads(request.data.decode("utf-8"))})
        return FakeResponse({"choices": [{"message": {"content": "{\"review\":\"ok\"}"}}]})

    client = LLMClient(
        LLMConfig(provider="deepseek", api_key="sk-test", model="deepseek-v4-flash", thinking=True),
        transport=fake_transport,
    )

    client.chat([{"role": "user", "content": "hello"}])

    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["body"]["thinking"] == {"type": "enabled"}
    assert calls[0]["body"]["reasoning_effort"] == "medium"


def test_llm_client_builds_anthropic_messages_payload_for_claude() -> None:
    calls: list[dict[str, object]] = []

    def fake_transport(request, timeout: float):
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeResponse({"content": [{"type": "text", "text": "{\"review\":\"ok\"}"}]})

    client = LLMClient(
        LLMConfig(provider="claude", api_key="sk-ant-test", model="claude-sonnet-4-5", max_tokens=2048),
        transport=fake_transport,
    )

    content = client.chat(
        [
            {"role": "system", "content": "system rule"},
            {"role": "user", "content": "evidence"},
        ]
    )

    assert content == "{\"review\":\"ok\"}"
    assert calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert calls[0]["headers"]["X-api-key"] == "sk-ant-test"
    assert calls[0]["headers"]["Anthropic-version"] == "2023-06-01"
    assert calls[0]["body"]["model"] == "claude-sonnet-4-5"
    assert calls[0]["body"]["system"] == "system rule"
    assert calls[0]["body"]["messages"] == [{"role": "user", "content": "evidence"}]
    assert calls[0]["body"]["max_tokens"] == 2048


def test_llm_client_uses_custom_base_url_without_double_appending_path() -> None:
    calls: list[str] = []

    def fake_transport(request, _timeout: float):
        calls.append(request.full_url)
        return FakeResponse({"choices": [{"message": {"content": "{\"review\":\"ok\"}"}}]})

    client = LLMClient(
        LLMConfig(
            provider="openai",
            api_key="sk-test",
            base_url="https://example.com/proxy/chat/completions",
            model="gpt-4.1",
        ),
        transport=fake_transport,
    )

    client.chat([{"role": "user", "content": "hello"}])

    assert calls == ["https://example.com/proxy/chat/completions"]


def test_llm_client_raises_for_missing_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(LLMConfig(provider="claude", api_key=""))

    with pytest.raises(LLMMissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        client.chat([{"role": "user", "content": "hello"}])


def test_llm_client_rejects_blank_base_url_before_http_request() -> None:
    def fake_transport(_request, _timeout: float):
        raise AssertionError("invalid base URL should not reach transport")

    client = LLMClient(LLMConfig(provider="openai", api_key="sk-test", base_url=" "), transport=fake_transport)

    with pytest.raises(LLMAPIError, match="Base URL"):
        client.chat([{"role": "user", "content": "hello"}])
