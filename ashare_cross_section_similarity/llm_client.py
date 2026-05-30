from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_LLM_PROVIDER = "deepseek"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class LLMAPIError(RuntimeError):
    pass


class LLMMissingAPIKeyError(LLMAPIError):
    pass


class LLMInvalidAPIKeyError(LLMAPIError):
    pass


@dataclass(frozen=True)
class LLMProviderPreset:
    provider: str
    label: str
    api_style: str
    default_base_url: str
    default_model: str
    api_key_env_var: str
    model_options: tuple[str, ...]
    supports_thinking: bool = False


def provider_presets() -> dict[str, LLMProviderPreset]:
    return {
        "deepseek": LLMProviderPreset(
            provider="deepseek",
            label="DeepSeek",
            api_style="openai_chat",
            default_base_url="https://api.deepseek.com",
            default_model="deepseek-v4-flash",
            api_key_env_var="DEEPSEEK_API_KEY",
            model_options=("deepseek-v4-flash", "deepseek-v4-pro"),
            supports_thinking=True,
        ),
        "qwen": LLMProviderPreset(
            provider="qwen",
            label="Qwen",
            api_style="openai_chat",
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            default_model="qwen-plus",
            api_key_env_var="DASHSCOPE_API_KEY",
            model_options=("qwen-plus", "qwen-max", "qwen-turbo"),
        ),
        "openai": LLMProviderPreset(
            provider="openai",
            label="OpenAI",
            api_style="openai_chat",
            default_base_url="https://api.openai.com/v1",
            default_model="gpt-4.1",
            api_key_env_var="OPENAI_API_KEY",
            model_options=("gpt-4.1", "gpt-4o", "gpt-4o-mini"),
        ),
        "claude": LLMProviderPreset(
            provider="claude",
            label="Claude",
            api_style="anthropic_messages",
            default_base_url="https://api.anthropic.com",
            default_model="claude-sonnet-4-5",
            api_key_env_var="ANTHROPIC_API_KEY",
            model_options=("claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"),
        ),
    }


def get_provider_preset(provider: str) -> LLMProviderPreset:
    key = _normalize_provider(provider)
    try:
        return provider_presets()[key]
    except KeyError as exc:
        choices = "、".join(preset.label for preset in provider_presets().values())
        raise LLMAPIError(f"未知 AI 供应商：{provider}。可选：{choices}。") from exc


@dataclass(frozen=True)
class LLMConfig:
    provider: str = DEFAULT_LLM_PROVIDER
    api_key: str | None = None
    base_url: str = ""
    model: str = ""
    timeout: float = 60.0
    thinking: bool = True
    reasoning_effort: str = "medium"
    max_tokens: int = 4096

    @property
    def preset(self) -> LLMProviderPreset:
        return get_provider_preset(self.provider)

    def resolved_base_url(self) -> str:
        base_url = (self.base_url or self.preset.default_base_url).strip().rstrip("/")
        if not base_url:
            raise LLMAPIError("Base URL 不能为空。")
        if not re.match(r"^https?://", base_url):
            raise LLMAPIError("Base URL 必须以 http:// 或 https:// 开头。")
        return base_url

    def resolved_model(self) -> str:
        model = (self.model or self.preset.default_model).strip()
        if not model:
            raise LLMAPIError("模型名不能为空。")
        return model

    def resolved_api_key(self) -> str:
        env_var = self.preset.api_key_env_var
        key = (self.api_key or os.environ.get(env_var, "")).strip()
        if not key:
            raise LLMMissingAPIKeyError(f"缺少 {env_var}，请设置环境变量或在页面临时输入 API Key。")
        if not key.isascii() or any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise LLMInvalidAPIKeyError("API Key 只能包含半角英文、数字和符号，请检查是否误填中文或空白字符。")
        return key


Transport = Callable[[Request, float], object]


def _default_transport(request: Request, timeout: float) -> object:
    return urlopen(request, timeout=timeout)


class LLMClient:
    def __init__(self, config: LLMConfig | None = None, *, transport: Transport | None = None) -> None:
        self.config = config or LLMConfig()
        self._transport = transport or _default_transport

    def chat(self, messages: list[dict[str, str]]) -> str:
        preset = self.config.preset
        if preset.api_style == "anthropic_messages":
            return self._anthropic_messages(messages)
        return self._openai_chat(messages)

    def _openai_chat(self, messages: list[dict[str, str]]) -> str:
        preset = self.config.preset
        payload: dict[str, object] = {
            "model": self.config.resolved_model(),
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if preset.supports_thinking:
            payload["thinking"] = {"type": "enabled" if self.config.thinking else "disabled"}
            payload["reasoning_effort"] = self.config.reasoning_effort
        request = Request(
            _chat_completions_url(self.config.resolved_base_url()),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.resolved_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        return _message_content(self._read_response(request))

    def _anthropic_messages(self, messages: list[dict[str, str]]) -> str:
        system, anthropic_messages = _anthropic_prompt_parts(messages)
        payload: dict[str, object] = {
            "model": self.config.resolved_model(),
            "max_tokens": int(self.config.max_tokens),
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        request = Request(
            _anthropic_messages_url(self.config.resolved_base_url()),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "x-api-key": self.config.resolved_api_key(),
                "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        return _anthropic_message_content(self._read_response(request))

    def _read_response(self, request: Request) -> str:
        provider_label = self.config.preset.label
        try:
            with self._transport(request, self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMAPIError(f"{provider_label} API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LLMAPIError(f"{provider_label} API 连接失败：{exc.reason}") from exc
        except OSError as exc:
            raise LLMAPIError(f"{provider_label} API 调用失败：{exc}") from exc
        return raw


def _normalize_provider(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    aliases = {
        "ds": "deepseek",
        "deepseek": "deepseek",
        "qwen": "qwen",
        "通义千问": "qwen",
        "oai": "openai",
        "openai": "openai",
        "claude": "claude",
        "anthropic": "claude",
    }
    return aliases.get(normalized, normalized)


def _chat_completions_url(base_url: str) -> str:
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


def _anthropic_messages_url(base_url: str) -> str:
    return base_url if base_url.endswith("/v1/messages") else f"{base_url}/v1/messages"


def _message_content(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAPIError(f"AI API 返回非 JSON：{exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAPIError("AI API 返回缺少 choices[0].message.content。") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMAPIError("AI API 返回 content 为空。")
    return content


def _anthropic_message_content(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMAPIError(f"Claude API 返回非 JSON：{exc}") from exc
    content = payload.get("content")
    if not isinstance(content, list):
        raise LLMAPIError("Claude API 返回缺少 content 数组。")
    parts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    text = "\n".join(part.strip() for part in parts if part.strip()).strip()
    if not text:
        raise LLMAPIError("Claude API 返回 content 为空。")
    return text


def _anthropic_prompt_parts(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    converted: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
            continue
        converted.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    if not converted:
        converted.append({"role": "user", "content": "请根据系统要求输出 JSON。"})
    return "\n\n".join(system_parts), converted
