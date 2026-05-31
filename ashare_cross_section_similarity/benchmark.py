from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ashare_cross_section_similarity.data import load_local_bars
from ashare_cross_section_similarity.features import normalized_close_path
from ashare_cross_section_similarity.history import HistorySearchConfig, HistorySearchResult, search_history
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionSearchResult,
    search_cross_section,
)
from ashare_cross_section_similarity.similarity_algorithms import get_algorithm_status
from ashare_cross_section_similarity.universe import unique_symbols


@dataclass(frozen=True)
class BenchmarkRunResult:
    summary: pd.DataFrame
    results: pd.DataFrame
    gallery_path: Path


def run_benchmark(
    *,
    cases_path: str | Path,
    algorithms: tuple[str, ...],
    output_dir: str | Path,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
) -> BenchmarkRunResult:
    cases = _load_cases(Path(cases_path))
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    gallery_sections: list[str] = []

    for case in cases:
        for algorithm in algorithms:
            status = get_algorithm_status(algorithm)
            if not status.available:
                summaries.append(
                    {
                        "case": case["name"],
                        "mode": case["mode"],
                        "algorithm": algorithm,
                        "status": "skipped",
                        "reason": status.reason,
                        "runtime_seconds": 0.0,
                        "result_count": 0,
                        "skipped_count": 0,
                    }
                )
                continue
            started = time.perf_counter()
            try:
                frame, section, skipped_count = _run_case(
                    case=case,
                    algorithm=algorithm,
                    data_root=Path(data_root),
                    timeframe=timeframe,
                    adjust=adjust,
                )
                runtime = time.perf_counter() - started
                summaries.append(
                    {
                        "case": case["name"],
                        "mode": case["mode"],
                        "algorithm": algorithm,
                        "status": "ok",
                        "reason": "",
                        "runtime_seconds": runtime,
                        "result_count": len(frame),
                        "skipped_count": skipped_count,
                    }
                )
                if not frame.empty:
                    frame.insert(0, "algorithm", algorithm)
                    frame.insert(0, "mode", case["mode"])
                    frame.insert(0, "case", case["name"])
                    result_frames.append(frame)
                gallery_sections.append(section)
            except Exception as exc:  # noqa: BLE001
                runtime = time.perf_counter() - started
                summaries.append(
                    {
                        "case": case["name"],
                        "mode": case["mode"],
                        "algorithm": algorithm,
                        "status": "failed",
                        "reason": str(exc),
                        "runtime_seconds": runtime,
                        "result_count": 0,
                        "skipped_count": 0,
                    }
                )

    summary = pd.DataFrame(summaries)
    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    results.to_csv(output / "benchmark_results.csv", index=False)
    gallery_path = output / "benchmark_gallery.html"
    gallery_path.write_text(_gallery_document(gallery_sections, summary), encoding="utf-8")
    return BenchmarkRunResult(summary=summary, results=results, gallery_path=gallery_path)


def _run_case(
    *,
    case: dict[str, object],
    algorithm: str,
    data_root: Path,
    timeframe: str,
    adjust: str,
) -> tuple[pd.DataFrame, str, int]:
    mode = str(case["mode"])
    if mode == "history":
        return _run_history_case(case, algorithm, data_root, timeframe, adjust)
    if mode == "cross_section":
        return _run_cross_section_case(case, algorithm, data_root, timeframe, adjust)
    raise ValueError(f"未知 benchmark mode：{mode}")


def _run_history_case(
    case: dict[str, object],
    algorithm: str,
    data_root: Path,
    timeframe: str,
    adjust: str,
) -> tuple[pd.DataFrame, str, int]:
    symbol = str(case["symbol"])
    start = str(case.get("start", "1900-01-01"))
    end = str(case["end"])
    top_n = int(case.get("top_n", 10))
    bars = load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=[symbol],
        start="1900-01-01",
        end=end,
    )
    selected = bars.loc[bars["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    result = search_history(
        bars,
        HistorySearchConfig(
            symbol=symbol,
            as_of=end,
            window_start=start,
            window_size=max(2, len(selected)),
            top_n=top_n,
            candidate_n=max(100, top_n),
            nearby_gap_days=int(case.get("nearby_gap_days", 0)),
            exclusion_bars=int(case.get("exclusion_bars", 0)),
            algorithm=algorithm,
        ),
    )
    section = _history_gallery_section(str(case["name"]), algorithm, result)
    return result.results.copy(), section, 0


def _run_cross_section_case(
    case: dict[str, object],
    algorithm: str,
    data_root: Path,
    timeframe: str,
    adjust: str,
) -> tuple[pd.DataFrame, str, int]:
    target_symbol = str(case["target_symbol"])
    universe = unique_symbols(case.get("universe_symbols", []))
    start = str(case["start"])
    end = str(case["end"])
    top_n = int(case.get("top_n", 20))
    tolerance = int(case.get("date_tolerance_bars", 0))
    symbols = unique_symbols([target_symbol, *universe])
    bars = load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=_benchmark_cross_section_load_end(end, tolerance),
    )
    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol=target_symbol,
            universe_symbols=tuple(universe),
            start=start,
            end=end,
            top_n=top_n,
            date_tolerance_bars=tolerance,
            algorithm=algorithm,
        ),
    )
    section = _cross_section_gallery_section(str(case["name"]), algorithm, bars, result)
    return result.results.copy(), section, len(result.skipped)


def _history_gallery_section(case_name: str, algorithm: str, result: HistorySearchResult) -> str:
    cards = [_window_card("当前窗口", result.current_window)]
    for index, window in enumerate(result.historical_windows[:6], start=1):
        cards.append(_window_card(f"样本{index}", window))
    return _section(case_name, algorithm, cards)


def _cross_section_gallery_section(
    case_name: str,
    algorithm: str,
    bars: pd.DataFrame,
    result: CrossSectionSearchResult,
) -> str:
    target = bars.loc[
        (bars["stock_code"] == result.target_symbol)
        & bars["date"].between(result.start, result.end)
    ]
    cards = [_window_card(f"{result.target_symbol} 目标", target)]
    for _, row in result.results.head(6).iterrows():
        symbol = str(row["symbol"])
        window = bars.loc[
            (bars["stock_code"] == symbol)
            & bars["date"].between(pd.Timestamp(row["区间开始"]), pd.Timestamp(row["区间结束"]))
        ]
        cards.append(_window_card(symbol, window))
    return _section(case_name, algorithm, cards)


def _section(case_name: str, algorithm: str, cards: list[str]) -> str:
    return (
        f"<section><h2>{escape(case_name)} / {escape(algorithm)}</h2>"
        f"<div class='grid'>{''.join(cards)}</div></section>"
    )


def _window_card(title: str, window: pd.DataFrame) -> str:
    start = window["date"].min() if not window.empty else ""
    end = window["date"].max() if not window.empty else ""
    return (
        "<article class='card'>"
        f"<h3>{escape(title)}</h3>"
        f"{_path_svg(window)}"
        f"<p>{escape(_date_text(start))} 至 {escape(_date_text(end))} | {len(window)} 根</p>"
        "</article>"
    )


def _path_svg(window: pd.DataFrame) -> str:
    if window.empty:
        return "<svg viewBox='0 0 240 120'></svg>"
    path = normalized_close_path(window)
    if len(path) == 0:
        return "<svg viewBox='0 0 240 120'></svg>"
    values = np.asarray(path, dtype=float)
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    span = max(max_value - min_value, 1e-9)
    points = []
    for index, value in enumerate(values):
        x = 8 + index * (224 / max(1, len(values) - 1))
        y = 110 - ((float(value) - min_value) / span) * 96
        points.append(f"{x:.1f},{y:.1f}")
    return (
        "<svg viewBox='0 0 240 120' aria-hidden='true'>"
        "<rect x='0' y='0' width='240' height='120' fill='#f8fafc'/>"
        f"<polyline points='{' '.join(points)}' fill='none' stroke='#0f6bff' stroke-width='2'/>"
        "</svg>"
    )


def _gallery_document(sections: list[str], summary: pd.DataFrame) -> str:
    table = summary.to_html(index=False, escape=True)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>相似算法核验图集</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #d7dde8; border-radius: 8px; padding: 12px; }}
.card h3 {{ margin: 0 0 8px; font-size: 14px; }}
.card p {{ margin: 8px 0 0; color: #4b5563; font-size: 12px; }}
table {{ border-collapse: collapse; margin-bottom: 28px; font-size: 13px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 640px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>相似算法核验图集</h1>
{table}
{''.join(sections)}
</body>
</html>
"""


def _date_text(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return str(value)
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _benchmark_cross_section_load_end(end: str | pd.Timestamp, date_tolerance_bars: int) -> str:
    extra_days = 0 if date_tolerance_bars <= 0 else max(date_tolerance_bars + 2, int(np.ceil(date_tolerance_bars * 2.2)))
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=extra_days + 45)
    return end_ts.strftime("%Y-%m-%d")


def _load_cases(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        import json

        payload = json.loads(text)
        return [dict(item) for item in payload.get("cases", [])]
    return _parse_simple_cases_yaml(text)


def _parse_simple_cases_yaml(text: str) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    list_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.rstrip()
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "cases:":
            continue
        if stripped.startswith("- ") and list_key and current is not None and indent > 2:
            current.setdefault(list_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if stripped.startswith("- "):
            if current is not None:
                cases.append(current)
            current = {}
            list_key = None
            remainder = stripped[2:].strip()
            if remainder:
                key, value = _split_yaml_key_value(remainder)
                current[key] = _parse_scalar(value)
            continue
        if current is None:
            continue
        key, value = _split_yaml_key_value(stripped)
        if value == "":
            current[key] = []
            list_key = key
        else:
            current[key] = _parse_scalar(value)
            list_key = None
    if current is not None:
        cases.append(current)
    return cases


def _split_yaml_key_value(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"无法解析 benchmark cases 行：{line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> object:
    value = value.strip().strip("'\"")
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value
