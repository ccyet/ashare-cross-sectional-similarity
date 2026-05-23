from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekAPIError(RuntimeError):
    pass


class DeepSeekMissingAPIKeyError(DeepSeekAPIError):
    pass


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEFAULT_DEEPSEEK_MODEL
    timeout: float = 60.0
    thinking: bool = True
    reasoning_effort: str = "medium"

    def resolved_api_key(self) -> str:
        key = (self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise DeepSeekMissingAPIKeyError("缺少 DEEPSEEK_API_KEY，请设置环境变量或在页面临时输入 API Key。")
        return key


Transport = Callable[[Request, float], object]


def _default_transport(request: Request, timeout: float) -> object:
    return urlopen(request, timeout=timeout)


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig | None = None, *, transport: Transport | None = None) -> None:
        self.config = config or DeepSeekConfig()
        self._transport = transport or _default_transport

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if self.config.thinking else "disabled"},
            "reasoning_effort": self.config.reasoning_effort,
        }
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.resolved_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._transport(request, self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekAPIError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise DeepSeekAPIError(f"DeepSeek API 连接失败：{exc.reason}") from exc
        except OSError as exc:
            raise DeepSeekAPIError(f"DeepSeek API 调用失败：{exc}") from exc
        return _message_content(raw)


def _message_content(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DeepSeekAPIError(f"DeepSeek API 返回非 JSON：{exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekAPIError("DeepSeek API 返回缺少 choices[0].message.content。") from exc
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekAPIError("DeepSeek API 返回 content 为空。")
    return content
