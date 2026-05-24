from __future__ import annotations

import json

import pytest

import ashare_cross_section_similarity.deepseek_client as deepseek_client_module
from ashare_cross_section_similarity.deepseek_client import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekConfig,
    DeepSeekMissingAPIKeyError,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_deepseek_client_builds_openai_compatible_payload() -> None:
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

    client = DeepSeekClient(
        DeepSeekConfig(api_key="sk-test", model="deepseek-v4-flash", thinking=True, reasoning_effort="high"),
        transport=fake_transport,
    )

    content = client.chat(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
    )

    assert content == "{\"review\":\"ok\"}"
    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["timeout"] == 60.0
    assert calls[0]["body"]["model"] == "deepseek-v4-flash"
    assert calls[0]["body"]["thinking"] == {"type": "enabled"}
    assert calls[0]["body"]["reasoning_effort"] == "high"
    assert calls[0]["body"]["response_format"] == {"type": "json_object"}
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_deepseek_client_default_transport_passes_timeout_by_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_urlopen(request, *, timeout: float):
        calls.append(
            {
                "url": request.full_url,
                "timeout": timeout,
            }
        )
        return FakeResponse({"choices": [{"message": {"content": "{\"review\":\"ok\"}"}}]})

    monkeypatch.setattr(deepseek_client_module, "urlopen", fake_urlopen)

    client = DeepSeekClient(DeepSeekConfig(api_key="sk-test", timeout=12.5))

    assert client.chat([{"role": "user", "content": "hello"}]) == "{\"review\":\"ok\"}"
    assert calls == [{"url": "https://api.deepseek.com/chat/completions", "timeout": 12.5}]


def test_deepseek_client_raises_for_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = DeepSeekClient(DeepSeekConfig(api_key=""))

    with pytest.raises(DeepSeekMissingAPIKeyError, match="DEEPSEEK_API_KEY"):
        client.chat([{"role": "user", "content": "hello"}])


def test_deepseek_client_rejects_non_ascii_key_before_http_request() -> None:
    def fake_transport(_request, _timeout: float):
        raise AssertionError("invalid API key should not reach transport")

    client = DeepSeekClient(DeepSeekConfig(api_key="测试key"), transport=fake_transport)

    with pytest.raises(DeepSeekAPIError, match="API Key"):
        client.chat([{"role": "user", "content": "hello"}])


def test_deepseek_client_raises_for_missing_content() -> None:
    def fake_transport(_request, _timeout: float):
        return FakeResponse({"choices": [{"message": {}}]})

    client = DeepSeekClient(DeepSeekConfig(api_key="sk-test"), transport=fake_transport)

    with pytest.raises(DeepSeekAPIError, match="content"):
        client.chat([{"role": "user", "content": "hello"}])
