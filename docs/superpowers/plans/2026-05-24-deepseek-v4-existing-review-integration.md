# DeepSeek V4 Existing Review Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect DeepSeek V4 to the existing走势复盘 capability so the current复盘页面/模块 can produce model-generated复盘、分析、锐评 without adding a new page.

**Architecture:** Reuse the deterministic review implementation already present on branch `codex/review-tdx-script-ui`, then add a small DeepSeek client and an AI review layer around its evidence. Keep history and cross-section search untouched. Page code only orchestrates inputs, evidence display, and the three model output panels.

**Tech Stack:** Python 3.11+, pandas, Streamlit, Plotly, standard-library `urllib.request` for DeepSeek API calls, pytest, ruff. DeepSeek OpenAI-compatible endpoint: `https://api.deepseek.com`, default model `deepseek-v4-flash`.

---

## Source Facts

- Current branch: `codex/producing`.
- Existing deterministic review branch: `codex/review-tdx-script-ui`.
- Existing review files on that branch:
  - `ashare_cross_section_similarity/review.py`
  - `tests/test_review.py`
  - review-related functions inside `streamlit_app.py`.
- DeepSeek official docs as of 2026-05-24:
  - V4 model IDs include `deepseek-v4-flash` and `deepseek-v4-pro`.
  - OpenAI base URL is `https://api.deepseek.com`.
  - Chat API supports `thinking` and `reasoning_effort`.
  - Pricing page lists `deepseek-v4-flash` as the economical default.

## File Map

- Restore/create: `ashare_cross_section_similarity/review.py`
  - Deterministic price-action review, comparison stats, ranking, and current natural-language local review.
- Create: `ashare_cross_section_similarity/deepseek_client.py`
  - Minimal DeepSeek V4 HTTP client, explicit errors, no dependency on OpenAI SDK.
- Create: `ashare_cross_section_similarity/review_ai.py`
  - Evidence builder, prompt builder, JSON parser, and result model for复盘/分析/锐评.
- Modify: `ashare_cross_section_similarity/cli.py`
  - Add `review` command with evidence-only and DeepSeek-backed output.
- Modify: `streamlit_app.py`
  - Bring in existing review UI from `codex/review-tdx-script-ui` without overwriting current branch features.
  - Add DeepSeek V4 config and three output panels inside the existing review page.
- Restore/create: `tests/test_review.py`
  - Existing deterministic review tests from `codex/review-tdx-script-ui`.
- Create: `tests/test_deepseek_client.py`
  - Client payload, missing key, HTTP failure, response extraction.
- Create: `tests/test_review_ai.py`
  - Evidence JSON shape, prompt messages, parser success/failure.
- Modify: `tests/test_cli.py`
  - `review` command parsing.
- Modify: `tests/test_streamlit_charts.py`
  - Helper-level tests for AI review rendering/config; no real Streamlit API call.
- Modify: `README.md`
  - Add concise DeepSeek V4 setup and CLI/page usage.

---

### Task 1: Restore Existing Deterministic Review Baseline

**Files:**
- Create: `ashare_cross_section_similarity/review.py`
- Create: `tests/test_review.py`
- Modify: `streamlit_app.py`

- [ ] **Step 1: Restore deterministic review module and tests from the existing branch**

Run:

```bash
git restore --source codex/review-tdx-script-ui -- ashare_cross_section_similarity/review.py tests/test_review.py
```

Expected:

```text
A  ashare_cross_section_similarity/review.py
A  tests/test_review.py
```

- [ ] **Step 2: Run the restored deterministic tests**

Run:

```bash
pytest -q tests/test_review.py
```

Expected: tests either pass, or fail only on import names missing from current branch integration. If failures are unrelated to the restored module, stop and inspect before editing.

- [ ] **Step 3: Extract review UI reference from the existing branch**

Run:

```bash
git show codex/review-tdx-script-ui:streamlit_app.py > /tmp/ashare_review_streamlit_app.py
rg -n "from ashare_cross_section_similarity.review|REVIEW_|DEFAULT_ANALYSIS_INDEX_SYMBOLS|SCRIPT_BENCHMARK_SYMBOL|def _render_review_tab|def _review_|render_review_text|render_multi_review_text|走势复盘|锐评" /tmp/ashare_review_streamlit_app.py
```

Expected: line numbers for review imports, constants, `_render_review_tab`, multi-review helpers, charts, formatters, and quick-window helpers.

- [ ] **Step 4: Manually merge only review imports/constants into current `streamlit_app.py`**

Add imports near the existing local imports:

```python
from ashare_cross_section_similarity.review import (
    ReviewConfig,
    analyze_price_review,
    build_comparison_stats,
    build_equal_weight_series,
    build_video_script_profile,
    rank_review_results,
    render_multi_review_text,
    render_multi_video_script_text,
    render_review_text,
    render_video_script_cards_html,
)
```

Add constants near current top-level constants, preserving current branch values:

```python
DEFAULT_ANALYSIS_INDEX_SYMBOLS = ("000300.SH", "000852.SH", "399006.SZ")
REVIEW_MAX_TARGET_SYMBOLS = 20
SCRIPT_BENCHMARK_SYMBOL = "000300.SH"
```

- [ ] **Step 5: Manually merge the existing review tab entry without deleting current tabs**

Change current tab creation from:

```python
history_tab, cross_section_tab = st.tabs(["历史时序相似", "横截面相似"])
```

to:

```python
history_tab, cross_section_tab, review_tab = st.tabs(["历史时序相似", "横截面相似", "走势复盘"])
```

Then add:

```python
with review_tab:
    _render_review_tab(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        provider=provider,
        download_engine=download_engine,
    )
```

- [ ] **Step 6: Merge review helper functions from `/tmp/ashare_review_streamlit_app.py`**

Copy these function groups from `/tmp/ashare_review_streamlit_app.py` into current `streamlit_app.py`, keeping current branch functions when names already exist:

```text
_render_review_tab
_render_multi_review_output
_review_shared_comparison_frames
_review_comparison_data
_review_video_script_profile
_review_metric_items
_review_target_symbols
_set_review_quick_window
_review_quick_window_feedback
_review_script_data_start
_review_auto_etf_proxies_from_index
_review_etf_index_with_fallback
_top_etf_options
_etf_option_formatter
_format_etf_matches
_etf_name_map_from_index
_has_etf_like_symbol
_review_kline_chart
_review_relative_chart
_format_review_segments
_format_review_comparisons
_format_video_script_profiles
```

Do not replace current branch data download, archive manager, TDX default, cache fingerprint, or history/cross-section helpers.

- [ ] **Step 7: Run review and Streamlit helper tests**

Run:

```bash
pytest -q tests/test_review.py tests/test_streamlit_charts.py
```

Expected: restored review tests pass. If `tests/test_streamlit_charts.py` fails because the current branch has newer expectations, preserve current branch behavior and adjust only review-specific integration.

- [ ] **Step 8: Commit deterministic review integration**

Run:

```bash
git add ashare_cross_section_similarity/review.py tests/test_review.py streamlit_app.py
git commit -m "feat: restore price action review workbench"
```

Expected: one commit containing the existing deterministic review baseline.

---

### Task 2: Add DeepSeek V4 Client

**Files:**
- Create: `ashare_cross_section_similarity/deepseek_client.py`
- Create: `tests/test_deepseek_client.py`

- [ ] **Step 1: Write failing client tests**

Create `tests/test_deepseek_client.py`:

```python
from __future__ import annotations

import json

import pytest

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


def test_deepseek_client_raises_for_missing_key() -> None:
    client = DeepSeekClient(DeepSeekConfig(api_key=""))

    with pytest.raises(DeepSeekMissingAPIKeyError, match="DEEPSEEK_API_KEY"):
        client.chat([{"role": "user", "content": "hello"}])


def test_deepseek_client_raises_for_missing_content() -> None:
    def fake_transport(_request, _timeout: float):
        return FakeResponse({"choices": [{"message": {}}]})

    client = DeepSeekClient(DeepSeekConfig(api_key="sk-test"), transport=fake_transport)

    with pytest.raises(DeepSeekAPIError, match="content"):
        client.chat([{"role": "user", "content": "hello"}])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest -q tests/test_deepseek_client.py
```

Expected: FAIL because `ashare_cross_section_similarity.deepseek_client` does not exist.

- [ ] **Step 3: Implement the minimal DeepSeek client**

Create `ashare_cross_section_similarity/deepseek_client.py`:

```python
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


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig | None = None, *, transport: Transport | None = None) -> None:
        self.config = config or DeepSeekConfig()
        self._transport = transport or urlopen

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
```

- [ ] **Step 4: Run client tests**

Run:

```bash
pytest -q tests/test_deepseek_client.py
```

Expected: PASS.

- [ ] **Step 5: Commit client**

Run:

```bash
git add ashare_cross_section_similarity/deepseek_client.py tests/test_deepseek_client.py
git commit -m "feat: add deepseek v4 client"
```

Expected: one commit for the client.

---

### Task 3: Build AI Review Evidence, Prompt, and Parser

**Files:**
- Create: `ashare_cross_section_similarity/review_ai.py`
- Create: `tests/test_review_ai.py`

- [ ] **Step 1: Write failing evidence and parser tests**

Create `tests/test_review_ai.py`:

```python
from __future__ import annotations

import json

import pandas as pd
import pytest

from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review, build_comparison_stats
from ashare_cross_section_similarity.review_ai import (
    ReviewAIFormatError,
    build_review_ai_evidence,
    build_review_ai_messages,
    parse_review_ai_result,
)


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "stock_code": [symbol] * len(closes),
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [100] * len(closes),
            "amount": [1000] * len(closes),
        }
    )


def test_build_review_ai_evidence_keeps_review_analysis_critique_inputs() -> None:
    target = _bars("000001.SZ", [10, 11, 12, 11, 13])
    result = analyze_price_review(target, ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-05"))
    comparisons = pd.DataFrame([build_comparison_stats(target, _bars("000300.SH", [10, 10.5, 10.4, 10.6, 10.7]), "沪深300")])

    evidence = build_review_ai_evidence(result, comparisons, stock_names={"000001.SZ": "平安银行"}, warnings=["样例风险"])

    assert evidence["target"]["symbol"] == "000001.SZ"
    assert evidence["target"]["name"] == "平安银行"
    assert evidence["overview"]["return"] == result.overview["return"]
    assert evidence["segments"]
    assert evidence["comparisons"][0]["标的"] == "沪深300"
    assert evidence["warnings"] == ["样例风险"]


def test_build_review_ai_messages_require_json_contract() -> None:
    messages = build_review_ai_messages({"target": {"symbol": "000001.SZ"}, "warnings": []})

    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert "review" in messages[0]["content"]
    assert "analysis" in messages[0]["content"]
    assert "critique" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "000001.SZ" in messages[1]["content"]


def test_parse_review_ai_result_accepts_required_fields() -> None:
    raw = json.dumps(
        {
            "review": "复盘内容",
            "analysis": "分析内容",
            "critique": "锐评内容",
            "evidence_refs": ["segments[0]", "comparisons[0]"],
            "disclaimer": "仅供研究复盘",
        },
        ensure_ascii=False,
    )

    result = parse_review_ai_result(raw)

    assert result.review == "复盘内容"
    assert result.analysis == "分析内容"
    assert result.critique == "锐评内容"
    assert result.evidence_refs == ("segments[0]", "comparisons[0]")


def test_parse_review_ai_result_rejects_missing_fields() -> None:
    with pytest.raises(ReviewAIFormatError, match="critique"):
        parse_review_ai_result('{"review":"ok","analysis":"ok"}')
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest -q tests/test_review_ai.py
```

Expected: FAIL because `review_ai.py` does not exist.

- [ ] **Step 3: Implement evidence, messages, and parser**

Create `ashare_cross_section_similarity/review_ai.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.review import ReviewResult


class ReviewAIFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ReviewAIResult:
    review: str
    analysis: str
    critique: str
    evidence_refs: tuple[str, ...]
    disclaimer: str
    raw: str


def build_review_ai_evidence(
    result: ReviewResult,
    comparisons: pd.DataFrame | None = None,
    *,
    stock_names: dict[str, str] | None = None,
    warnings: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    names = stock_names or {}
    symbol = result.symbol
    return {
        "target": {
            "symbol": symbol,
            "name": names.get(symbol, ""),
            "start": _date_text(result.start),
            "end": _date_text(result.end),
            "bars": int(len(result.window)),
        },
        "overview": _json_safe_mapping(result.overview),
        "segments": _frame_records(result.main_segments),
        "comparisons": _frame_records(comparisons if comparisons is not None else pd.DataFrame()),
        "warnings": [str(item) for item in warnings if str(item).strip()],
        "limits": [
            "只基于本地行情、相似度和对比统计，不读取新闻或基本面。",
            "输出仅用于研究复盘，不构成投资建议。",
        ],
    }


def build_review_ai_messages(evidence: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是A股走势复盘助手。必须只基于用户提供的JSON证据做复盘、分析、锐评，"
        "不得编造新闻、基本面、资金流或未提供的数据。"
        "输出必须是严格JSON对象，字段只能包含：review、analysis、critique、evidence_refs、disclaimer。"
        "review写结构化复盘；analysis写数据分析；critique写锐评和反证；"
        "evidence_refs列出引用的证据字段，例如 segments[0] 或 comparisons[0]。"
    )
    user = json.dumps(evidence, ensure_ascii=False, default=str)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_review_ai_result(raw: str) -> ReviewAIResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewAIFormatError(f"模型输出不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewAIFormatError("模型输出必须是 JSON 对象。")
    missing = [field for field in ("review", "analysis", "critique") if not str(payload.get(field, "")).strip()]
    if missing:
        raise ReviewAIFormatError(f"模型输出缺少必要字段：{', '.join(missing)}。")
    refs = payload.get("evidence_refs", [])
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        raise ReviewAIFormatError("evidence_refs 必须是字符串数组。")
    return ReviewAIResult(
        review=str(payload["review"]).strip(),
        analysis=str(payload["analysis"]).strip(),
        critique=str(payload["critique"]).strip(),
        evidence_refs=tuple(str(item).strip() for item in refs if str(item).strip()),
        disclaimer=str(payload.get("disclaimer") or "仅用于研究复盘，不构成投资建议。").strip(),
        raw=raw,
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [_json_safe_mapping(row) for row in frame.to_dict(orient="records")]


def _json_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in values.items():
        if pd.isna(value):
            safe[str(key)] = None
        elif isinstance(value, pd.Timestamp):
            safe[str(key)] = _date_text(value)
        else:
            safe[str(key)] = value.item() if hasattr(value, "item") else value
    return safe


def _date_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")
```

- [ ] **Step 4: Run AI review tests**

Run:

```bash
pytest -q tests/test_review_ai.py
```

Expected: PASS.

- [ ] **Step 5: Commit AI review layer**

Run:

```bash
git add ashare_cross_section_similarity/review_ai.py tests/test_review_ai.py
git commit -m "feat: build ai review evidence contract"
```

Expected: one commit for evidence/prompt/parser.

---

### Task 4: Add CLI `review` Command

**Files:**
- Modify: `ashare_cross_section_similarity/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI parser test**

Append to `tests/test_cli.py`:

```python
def test_parse_review_command() -> None:
    args = _parse_args(
        [
            "review",
            "--target-symbol",
            "300750.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--model",
            "deepseek-v4-flash",
            "--output",
            "outputs/review.json",
        ]
    )

    assert args.command == "review"
    assert args.target_symbol == "300750.SZ"
    assert args.model == "deepseek-v4-flash"
    assert args.output == "outputs/review.json"
```

- [ ] **Step 2: Run parser test to verify failure**

Run:

```bash
pytest -q tests/test_cli.py::test_parse_review_command
```

Expected: FAIL because `review` is not an accepted command.

- [ ] **Step 3: Implement review command parser and runner**

Modify `ashare_cross_section_similarity/cli.py`:

Add imports:

```python
import json
import os

from ashare_cross_section_similarity.deepseek_client import DeepSeekClient, DeepSeekConfig
from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review
from ashare_cross_section_similarity.review_ai import (
    build_review_ai_evidence,
    build_review_ai_messages,
    parse_review_ai_result,
)
```

Update command dispatch:

```python
if args.command == "review":
    return _run_review(args)
```

Update implicit command list:

```python
if argv and argv[0] not in {"search", "history", "review", "download", "check", "import-data", "benchmark", "-h", "--help"}:
    argv.insert(0, "search")
```

Add parser:

```python
review_parser = subparsers.add_parser("review", help="生成走势复盘、分析和锐评")
_add_common_data_args(review_parser)
review_parser.add_argument("--target-symbol", required=True, help="目标代码，如 300750.SZ")
review_parser.add_argument("--start", required=True, help="复盘区间开始")
review_parser.add_argument("--end", required=True, help="复盘区间结束")
review_parser.add_argument("--model", default="deepseek-v4-flash", help="DeepSeek V4 模型名")
review_parser.add_argument("--api-key", default="", help="DeepSeek API Key；留空读取 DEEPSEEK_API_KEY")
review_parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek OpenAI-compatible Base URL")
review_parser.add_argument("--no-thinking", action="store_true", help="关闭 DeepSeek thinking")
review_parser.add_argument("--evidence-only", action="store_true", help="只输出证据包，不调用 DeepSeek")
review_parser.add_argument("--output", default="", help="JSON 输出路径")
```

Add runner:

```python
def _run_review(args: argparse.Namespace) -> int:
    bars = load_local_bars(
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        symbols=[args.target_symbol],
        start=args.start,
        end=args.end,
    )
    result = analyze_price_review(
        bars,
        ReviewConfig(symbol=args.target_symbol, start=args.start, end=args.end),
    )
    evidence = build_review_ai_evidence(result, pd.DataFrame(), warnings=list(result.warnings))
    payload: dict[str, object] = {"evidence": evidence}
    if not args.evidence_only:
        client = DeepSeekClient(
            DeepSeekConfig(
                api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url=args.base_url,
                model=args.model,
                thinking=not bool(args.no_thinking),
            )
        )
        ai_result = parse_review_ai_result(client.chat(build_review_ai_messages(evidence)))
        payload["ai_review"] = {
            "review": ai_result.review,
            "analysis": ai_result.analysis,
            "critique": ai_result.critique,
            "evidence_refs": list(ai_result.evidence_refs),
            "disclaimer": ai_result.disclaimer,
        }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"复盘结果已写入：{output_path}")
    else:
        print(text)
    return 0
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest -q tests/test_cli.py::test_parse_review_command
```

Expected: PASS.

- [ ] **Step 5: Smoke test evidence-only CLI with local synthetic data path if available**

Run:

```bash
python -m ashare_cross_section_similarity review \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --timeframe 1d \
  --target-symbol 000852.SH \
  --start 2026-01-01 \
  --end 2026-05-15 \
  --evidence-only \
  --output outputs/review_evidence_smoke.json
```

Expected: command returns 0 and writes `outputs/review_evidence_smoke.json`. If local data is absent, record the explicit data error and continue with unit-level verification.

- [ ] **Step 6: Commit CLI command**

Run:

```bash
git add ashare_cross_section_similarity/cli.py tests/test_cli.py
git commit -m "feat: add review cli command"
```

Expected: one commit for CLI.

---

### Task 5: Connect DeepSeek V4 to Existing Review Page

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_streamlit_charts.py`

- [ ] **Step 1: Add helper tests for rendering AI result**

Append to `tests/test_streamlit_charts.py`:

```python
from ashare_cross_section_similarity.review_ai import ReviewAIResult
from streamlit_app import _review_ai_display_sections


def test_review_ai_display_sections_returns_review_analysis_critique() -> None:
    result = ReviewAIResult(
        review="复盘",
        analysis="分析",
        critique="锐评",
        evidence_refs=("segments[0]",),
        disclaimer="仅供研究",
        raw="{}",
    )

    sections = _review_ai_display_sections(result)

    assert sections == [("复盘", "复盘"), ("分析", "分析"), ("锐评", "锐评")]
```

- [ ] **Step 2: Run helper test to verify failure**

Run:

```bash
pytest -q tests/test_streamlit_charts.py::test_review_ai_display_sections_returns_review_analysis_critique
```

Expected: FAIL because `_review_ai_display_sections` does not exist.

- [ ] **Step 3: Add imports and helper functions to `streamlit_app.py`**

Add imports:

```python
from ashare_cross_section_similarity.deepseek_client import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekConfig,
    DEFAULT_DEEPSEEK_MODEL,
)
from ashare_cross_section_similarity.review_ai import (
    ReviewAIFormatError,
    ReviewAIResult,
    build_review_ai_evidence,
    build_review_ai_messages,
    parse_review_ai_result,
)
```

Add helper functions near other review helpers:

```python
def _review_ai_display_sections(result: ReviewAIResult) -> list[tuple[str, str]]:
    return [("复盘", result.review), ("分析", result.analysis), ("锐评", result.critique)]


def _render_review_ai_result(result: ReviewAIResult) -> None:
    for column, (title, body) in zip(st.columns(3), _review_ai_display_sections(result)):
        with column:
            st.markdown(f"**{title}**")
            st.markdown(body)
    if result.evidence_refs:
        st.caption("证据引用：" + "、".join(result.evidence_refs))
    st.caption(result.disclaimer)
```

- [ ] **Step 4: Add DeepSeek controls inside existing `_render_review_tab`**

Inside `_render_review_tab`, after deterministic `st.markdown("**3. 自然语言复盘**")` output and before detail tables, add:

```python
    st.markdown("**4. DeepSeek V4 复盘 / 分析 / 锐评**")
    ai_col1, ai_col2, ai_col3 = st.columns([1.2, 1.2, 1])
    deepseek_model = ai_col1.selectbox(
        "模型",
        ["deepseek-v4-flash", "deepseek-v4-pro"],
        index=0,
        key="review_ai_model",
    )
    deepseek_key = ai_col2.text_input(
        "DeepSeek API Key",
        value="",
        type="password",
        key="review_ai_api_key",
        help="留空时读取环境变量 DEEPSEEK_API_KEY。",
    )
    deepseek_thinking = ai_col3.checkbox("启用 Thinking", value=True, key="review_ai_thinking")
    evidence = build_review_ai_evidence(
        result,
        comparison_frame,
        stock_names=stock_names,
        warnings=all_warnings,
    )
    with st.expander("查看发送给 DeepSeek 的证据摘要"):
        st.json(evidence)
    if st.button("生成 DeepSeek 复盘/分析/锐评", type="secondary", key="review_ai_run"):
        try:
            client = DeepSeekClient(
                DeepSeekConfig(
                    api_key=deepseek_key,
                    model=str(deepseek_model),
                    thinking=bool(deepseek_thinking),
                )
            )
            ai_result = parse_review_ai_result(client.chat(build_review_ai_messages(evidence)))
        except (DeepSeekAPIError, ReviewAIFormatError) as exc:
            st.error(str(exc))
        else:
            st.session_state["review_ai_result"] = ai_result
    if st.session_state.get("review_ai_result"):
        _render_review_ai_result(st.session_state["review_ai_result"])
```

If the copied review tab already uses section numbers `4` for detail tables, renumber detail tables to `**5. 波段与对比明细**`.

- [ ] **Step 5: Add equivalent multi-stock AI hook**

Inside `_render_multi_review_output`, after deterministic multi natural-language review output, build a compact evidence object:

```python
    multi_evidence = {
        "mode": "multi_stock",
        "targets": [result.symbol for result in results],
        "rankings": _frame_records(rank_review_results(results, comparison_frame, stock_names=stock_names)),
        "comparisons": _frame_records(comparison_frame),
        "warnings": all_warnings,
        "limits": ["只基于本地行情和对比统计，不读取新闻或基本面。"],
    }
```

Use the same DeepSeek controls and `_render_review_ai_result`. If `_frame_records` is private inside `review_ai.py`, expose it as `frame_records_for_ai(frame: pd.DataFrame)`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest -q tests/test_review.py tests/test_review_ai.py tests/test_deepseek_client.py tests/test_streamlit_charts.py::test_review_ai_display_sections_returns_review_analysis_critique
```

Expected: PASS.

- [ ] **Step 7: Commit Streamlit integration**

Run:

```bash
git add streamlit_app.py tests/test_streamlit_charts.py
git commit -m "feat: connect deepseek to review workbench"
```

Expected: one commit for page integration.

---

### Task 6: UI Polish for Product Workbench Feel

**Files:**
- Modify: `streamlit_app.py`
- Modify: `tests/test_streamlit_charts.py`

- [ ] **Step 1: Add helper test for CSS containing workbench colors**

Append to `tests/test_streamlit_charts.py`:

```python
from streamlit_app import _app_theme_css


def test_app_theme_css_uses_workbench_palette() -> None:
    css = _app_theme_css()

    assert "#2563eb" in css
    assert "#111827" in css
    assert "linear-gradient" not in css
```

- [ ] **Step 2: Run CSS test to verify failure**

Run:

```bash
pytest -q tests/test_streamlit_charts.py::test_app_theme_css_uses_workbench_palette
```

Expected: FAIL because `_app_theme_css` does not exist.

- [ ] **Step 3: Add restrained CSS helper and inject once**

Add near top-level helpers:

```python
def _app_theme_css() -> str:
    return """
<style>
:root {
  --ashare-text: #111827;
  --ashare-muted: #6b7280;
  --ashare-line: #e5e7eb;
  --ashare-blue: #2563eb;
  --ashare-bg: #f8fafc;
}
.stApp {
  background: var(--ashare-bg);
  color: var(--ashare-text);
}
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--ashare-line);
  border-radius: 8px;
  padding: 10px 12px;
}
.stButton > button[kind="primary"] {
  background: var(--ashare-blue);
  border-color: var(--ashare-blue);
}
</style>
"""
```

In `main()`, after `st.set_page_config(...)`, add:

```python
st.markdown(_app_theme_css(), unsafe_allow_html=True)
```

- [ ] **Step 4: Run CSS test**

Run:

```bash
pytest -q tests/test_streamlit_charts.py::test_app_theme_css_uses_workbench_palette
```

Expected: PASS.

- [ ] **Step 5: Commit UI polish**

Run:

```bash
git add streamlit_app.py tests/test_streamlit_charts.py
git commit -m "style: tune research workbench theme"
```

Expected: one commit for UI polish.

---

### Task 7: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/handoff.md`

- [ ] **Step 1: Update README with DeepSeek setup**

Add a section after current page usage:

```markdown
## DeepSeek V4 复盘、分析和锐评

页面里的 `走势复盘` 使用本地 K 线先生成可复验的确定性复盘；配置 DeepSeek 后，可继续生成大模型版 `复盘 / 分析 / 锐评`。

```bash
export DEEPSEEK_API_KEY=your_api_key
streamlit run streamlit_app.py
```

CLI 可只生成证据包，也可调用 DeepSeek：

```bash
python -m ashare_cross_section_similarity review \
  --data-root /Users/a1234/Desktop/trend-backtest/data/market/daily \
  --target-symbol 000852.SH \
  --start 2026-01-01 \
  --end 2026-05-15 \
  --model deepseek-v4-flash \
  --output outputs/review.json
```

未设置 `DEEPSEEK_API_KEY` 时，程序会明确报错，不会伪造大模型结果。
```

- [ ] **Step 2: Update handoff**

Add to `docs/handoff.md` under current abilities:

```markdown
- `走势复盘`：复用已有本地 K 线复盘、指数/板块对比和排序锐评，并可选接入 DeepSeek V4 生成复盘、分析和锐评。API Key 从 `DEEPSEEK_API_KEY` 或页面会话输入读取，不写入源码。
```

- [ ] **Step 3: Run full verification**

Run:

```bash
pytest -q
ruff check .
python -m py_compile streamlit_app.py ashare_cross_section_similarity/*.py scripts/download_all_a_daily.py
git diff --check
```

Expected:

```text
pytest: all tests passed
ruff: All checks passed
py_compile: no output
git diff --check: no output
```

- [ ] **Step 4: Run local Streamlit preview**

Run:

```bash
streamlit run streamlit_app.py --server.port 8502
```

Then verify:

```bash
curl -sf http://localhost:8502/_stcore/health
```

Expected:

```text
ok
```

Open `http://localhost:8502/` in the local browser and verify:

```text
标题：A股相似阶段搜集
Tab：历史时序相似
Tab：横截面相似
Tab：走势复盘
走势复盘内可见：DeepSeek V4 复盘 / 分析 / 锐评
```

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md docs/handoff.md
git commit -m "docs: document deepseek review workflow"
```

Expected: final documentation commit.

---

## Self-Review Checklist

- Spec coverage:
  - Existing复盘 page/module is reused through Task 1 and Task 5.
  - DeepSeek V4 is fixed as first model through Task 2 and Task 5.
  - 复盘、分析、锐评 are represented in Task 3 and Task 5.
  - CLI and product packaging path are covered in Task 4 and Task 7.
  - UI workbench feel is covered in Task 6.
  - Explicit failure handling is covered by client/parser tests.
- Placeholder scan: no incomplete placeholder steps.
- Type consistency:
  - `ReviewAIResult.review/analysis/critique` are used consistently.
  - `DeepSeekConfig` fields match tests and page code.
  - `build_review_ai_evidence` accepts `ReviewResult`, comparison frame, stock names, warnings.
