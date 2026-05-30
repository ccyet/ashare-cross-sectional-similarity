from __future__ import annotations

from typing import Any

import pandas as pd

from ashare_cross_section_similarity.data import CANONICAL_COLUMNS
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols


def fetch_openbb_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    provider: str = "akshare",
    timeframe: str = "1d",
    obb_client: Any | None = None,
) -> pd.DataFrame:
    obb = obb_client or _load_obb()
    frames: list[pd.DataFrame] = []
    provider_name = provider or "akshare"
    for symbol in unique_symbols(symbols):
        output = obb.equity.price.historical(
            symbol=_provider_symbol(symbol, provider_name),
            start_date=start,
            end_date=end,
            interval=timeframe,
            provider=provider_name,
        )
        frame = _normalize_openbb_frame(_output_to_frame(output), symbol)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def _load_obb() -> Any:
    try:
        from openbb import obb
    except ImportError as exc:
        raise RuntimeError(
            "未安装 OpenBB 数据源。请先执行：pip install openbb openbb_akshare；"
            '如使用 AKShare 扩展，再执行：python -c "import openbb; openbb.build()"'
        ) from exc
    return obb


def _provider_symbol(symbol: str, provider: str) -> str:
    normalized = normalize_symbol(symbol)
    if provider.lower() in {"akshare", "tushare"} and "." in normalized:
        return normalized.split(".", 1)[0]
    return normalized


def _output_to_frame(output: Any) -> pd.DataFrame:
    if hasattr(output, "to_dataframe"):
        frame = output.to_dataframe()
    elif hasattr(output, "to_df"):
        frame = output.to_df()
    else:
        raise TypeError("OpenBB 返回对象缺少 to_dataframe/to_df 方法。")
    return pd.DataFrame(frame).reset_index()


def _normalize_openbb_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    result = frame.copy()
    result.columns = [str(column).lower() for column in result.columns]
    if "date" not in result.columns and "datetime" in result.columns:
        result = result.rename(columns={"datetime": "date"})
    if "date" not in result.columns and "index" in result.columns:
        result = result.rename(columns={"index": "date"})
    missing = [column for column in ("date", "open", "high", "low", "close") if column not in result]
    if missing:
        raise ValueError(f"OpenBB 返回缺少必要列：{', '.join(missing)}")
    result["stock_code"] = normalize_symbol(symbol)
    for column in ("volume", "amount"):
        if column not in result.columns:
            result[column] = pd.NA
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    return result[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)
