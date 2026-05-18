from __future__ import annotations

from collections.abc import Mapping
import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.data import CANONICAL_COLUMNS
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

TDX_TQCENTER_ENV_VAR = "TDX_TQCENTER_PATH"
TIMEFRAME_PERIODS = {"1d": "1d", "30m": "30m", "15m": "15m", "5m": "5m", "1m": "1m"}
ADJUST_MAP = {"": "none", "qfq": "front", "hfq": "back"}
REQUIRED_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount")
FIELD_ALIASES = {
    "Open": ("Open", "open"),
    "High": ("High", "high"),
    "Low": ("Low", "low"),
    "Close": ("Close", "close"),
    "Volume": ("Volume", "volume", "vol"),
    "Amount": ("Amount", "amount"),
}
OUTPUT_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
}

_TQ_CLIENT: Any | None = None
_INITIALIZED = False


def fetch_tdx_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    timeframe: str = "1d",
    adjust: str = "qfq",
    tqcenter_path: str = "",
    tq_client: Any | None = None,
) -> pd.DataFrame:
    period = TIMEFRAME_PERIODS.get(timeframe)
    if period is None:
        raise ValueError("timeframe 仅支持 1d、30m、15m、5m、1m。")
    dividend_type = ADJUST_MAP.get(adjust)
    if dividend_type is None:
        raise ValueError("adjust 仅支持 qfq、hfq 或空字符串。")

    tq = tq_client or _load_tq(tqcenter_path)
    _ensure_initialized(tq)
    frames: list[pd.DataFrame] = []
    for symbol in unique_symbols(symbols):
        payload = tq.get_market_data(
            field_list=list(REQUIRED_FIELDS),
            stock_list=[symbol],
            period=period,
            start_time=_format_market_time(start),
            end_time=_format_market_time(end),
            count=-1,
            dividend_type=dividend_type,
            fill_data=False,
        )
        frame = _normalize_tdx_payload(payload, symbol=symbol, start=start, end=end)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def _load_tq(tqcenter_path: str = "") -> Any:
    global _TQ_CLIENT
    if _TQ_CLIENT is not None:
        return _TQ_CLIENT

    errors: list[str] = []
    for path in _candidate_import_paths(tqcenter_path):
        resolved = path.resolve()
        if not resolved.exists():
            errors.append(f"{resolved} 不存在")
            continue
        inserted = False
        path_text = str(resolved)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted = True
        try:
            module = importlib.import_module("tqcenter")
            _TQ_CLIENT = getattr(module, "tq")
            return _TQ_CLIENT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_text}: {exc}")
            if inserted:
                try:
                    sys.path.remove(path_text)
                except ValueError:
                    pass

    try:
        module = importlib.import_module("tqcenter")
        _TQ_CLIENT = getattr(module, "tq")
        return _TQ_CLIENT
    except Exception as exc:  # noqa: BLE001
        errors.append(f"normal import: {exc}")
        details = " | ".join(errors)
        raise RuntimeError(
            "无法导入 tqcenter。请先安装并登录本机通达信终端，并通过"
            f" {TDX_TQCENTER_ENV_VAR} 或下载源输入框指向 TDX 的 PYPlugins/user 目录。"
            f" 详情: {details}"
        ) from exc


def _candidate_import_paths(tqcenter_path: str = "") -> list[Path]:
    raw_value = (tqcenter_path or os.getenv(TDX_TQCENTER_ENV_VAR, "")).strip()
    if not raw_value or raw_value.lower() == "tdx":
        return []

    def expand(candidate: Path) -> list[Path]:
        normalized = candidate
        if normalized.name.lower() == "tqcenter.py":
            normalized = normalized.parent
        if normalized.name.lower() == "user" and normalized.parent.name.lower() == "pyplugins":
            return [normalized]
        if normalized.name.lower() == "pyplugins":
            return [normalized / "user", normalized]
        return [normalized / "PYPlugins" / "user", normalized]

    paths: list[Path] = []
    seen: set[str] = set()
    for item in raw_value.split(os.pathsep):
        text = item.strip().strip('"')
        if not text:
            continue
        for path in expand(Path(text).expanduser()):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def _ensure_initialized(tq: Any) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    try:
        tq.initialize(__file__)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("TDX 初始化失败。请确认本机通达信终端已启动并登录。") from exc
    _INITIALIZED = True


def _format_market_time(value: str) -> str:
    parsed = pd.Timestamp(pd.to_datetime(value))
    return parsed.strftime("%Y%m%d%H%M%S" if _has_explicit_time(value) else "%Y%m%d")


def _has_explicit_time(value: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}", str(value).strip()))


def _filter_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(pd.to_datetime(start))
    end_ts = pd.Timestamp(pd.to_datetime(end))
    if not _has_explicit_time(start):
        start_ts = start_ts.normalize()
    if not _has_explicit_time(end):
        end_ts = end_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return start_ts, end_ts


def _normalize_tdx_payload(
    raw_data: Any,
    *,
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(symbol)
    if raw_data is None:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"TDX 返回应为 dict[field]->DataFrame，实际为 {type(raw_data).__name__}。")
    payload = raw_data
    if not payload:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    selected_frames: dict[str, pd.DataFrame] = {}
    for field in REQUIRED_FIELDS:
        key = _resolve_field_key(payload, field)
        if key is None:
            raise ValueError(f"TDX 返回缺少必要字段：{field}")
        value = payload[key]
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"TDX 字段 {key} 应为 DataFrame，实际为 {type(value).__name__}。")
        columns = {str(column): column for column in value.columns}
        if normalized_symbol not in columns:
            raise ValueError(f"missing symbol column {normalized_symbol} in field {key}")
        selected_frames[field] = value

    if all(frame.empty for frame in selected_frames.values()):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    series_map: dict[str, pd.Series] = {}
    for field, frame in selected_frames.items():
        column = {str(item): item for item in frame.columns}[normalized_symbol]
        series_map[field] = pd.Series(
            pd.to_numeric(frame[column].to_numpy(), errors="coerce"),
            index=pd.to_datetime(frame.index, errors="coerce"),
            name=field,
        )

    assembled = pd.concat(series_map, axis=1).reset_index().rename(columns={"index": "date"})
    assembled = assembled.rename(columns=OUTPUT_RENAME)
    assembled["date"] = pd.to_datetime(assembled["date"], errors="coerce")
    start_ts, end_ts = _filter_window(start, end)
    assembled = assembled.loc[assembled["date"].between(start_ts, end_ts)].copy()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        assembled[column] = pd.to_numeric(assembled[column], errors="coerce")
    assembled["stock_code"] = normalized_symbol
    assembled = assembled.dropna(subset=["date", "open", "high", "low", "close"])
    assembled = assembled[CANONICAL_COLUMNS].drop_duplicates(subset=["stock_code", "date"], keep="last")
    return assembled.sort_values("date").reset_index(drop=True)


def _resolve_field_key(payload: Mapping[str, Any], field: str) -> str | None:
    for key in FIELD_ALIASES[field]:
        if key in payload:
            return key
    return None
