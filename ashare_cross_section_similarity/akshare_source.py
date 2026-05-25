from __future__ import annotations

from typing import Any

import pandas as pd

from ashare_cross_section_similarity.data import CANONICAL_COLUMNS
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

AKSHARE_ADJUST_MAP = {"qfq": "qfq", "hfq": "hfq", "": ""}
AKSHARE_COLUMN_MAP = {
    "日期": "date",
    "时间": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}


def fetch_akshare_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    timeframe: str = "1d",
    adjust: str = "qfq",
    ak_client: Any | None = None,
) -> pd.DataFrame:
    if timeframe != "1d":
        raise ValueError("原生 AkShare 抓取当前仅支持 1d；分钟线请使用 TDX。")
    ak = ak_client or _load_akshare()
    adjust_value = AKSHARE_ADJUST_MAP.get(adjust)
    if adjust_value is None:
        raise ValueError("adjust 仅支持 qfq、hfq 或空字符串。")

    frames: list[pd.DataFrame] = []
    for symbol in unique_symbols(symbols):
        frame = _fetch_one_symbol(ak, symbol=symbol, start=start, end=end, adjust=adjust_value)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def _load_akshare() -> Any:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("未安装 AkShare 数据源。请先执行：pip install akshare") from exc
    return ak


def _fetch_one_symbol(ak: Any, *, symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    normalized = normalize_symbol(symbol)
    if _is_mainland_index(normalized):
        errors: list[str] = []
        for source_name, fetcher in _index_fetchers(ak, normalized, start=start, end=end):
            try:
                raw = fetcher()
                frame = _filter_date_range(_normalize_akshare_frame(pd.DataFrame(raw), normalized), start, end)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source_name}: {exc}")
                continue
            if not frame.empty:
                return frame
            errors.append(f"{source_name}: 空结果")
        raise RuntimeError(f"AkShare 未能获取 {normalized} 指数行情；已尝试东财、腾讯、新浪。详情：{' | '.join(errors)}")
    errors: list[str] = []
    for source_name, fetcher in _stock_fetchers(ak, normalized, start=start, end=end, adjust=adjust):
        try:
            raw = fetcher()
            frame = _filter_date_range(_normalize_akshare_frame(pd.DataFrame(raw), normalized), start, end)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source_name}: {exc}")
            continue
        if not frame.empty:
            return frame
        errors.append(f"{source_name}: 空结果")
    raise RuntimeError(f"AkShare 未能获取 {normalized} 行情；已尝试东财、腾讯、新浪。详情：{' | '.join(errors)}")


def _stock_fetchers(ak: Any, symbol: str, *, start: str, end: str, adjust: str) -> list[tuple[str, Any]]:
    code = symbol.split(".", 1)[0]
    prefixed = _market_prefixed_symbol(symbol)

    def eastmoney() -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=_akshare_date(start),
            end_date=_akshare_date(end),
            adjust=adjust,
        )

    def tencent() -> pd.DataFrame:
        return ak.stock_zh_a_hist_tx(
            symbol=prefixed,
            start_date=_akshare_date(start),
            end_date=_akshare_date(end),
            adjust=adjust,
        )

    def sina() -> pd.DataFrame:
        return ak.stock_zh_a_daily(
            symbol=prefixed,
            start_date=_akshare_date(start),
            end_date=_akshare_date(end),
            adjust=adjust,
        )

    return [("eastmoney", eastmoney), ("tencent", tencent), ("sina", sina)]


def _index_fetchers(ak: Any, symbol: str, *, start: str, end: str) -> list[tuple[str, Any]]:
    code = symbol.split(".", 1)[0]
    prefixed = _market_prefixed_symbol(symbol)

    def eastmoney() -> pd.DataFrame:
        return ak.stock_zh_index_daily_em(
            symbol=code,
            start_date=_akshare_date(start),
            end_date=_akshare_date(end),
        )

    def tencent() -> pd.DataFrame:
        return ak.stock_zh_index_daily_tx(symbol=prefixed)

    def sina() -> pd.DataFrame:
        return ak.stock_zh_index_daily(symbol=prefixed)

    return [("eastmoney", eastmoney), ("tencent", tencent), ("sina", sina)]


def _market_prefixed_symbol(symbol: str) -> str:
    code, _, suffix = normalize_symbol(symbol).partition(".")
    prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix, suffix.lower())
    return f"{prefix}{code}"


def _normalize_akshare_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    renamed = frame.rename(columns={column: AKSHARE_COLUMN_MAP.get(str(column), str(column)) for column in frame.columns})
    missing = [column for column in ("date", "open", "high", "low", "close") if column not in renamed.columns]
    if missing:
        raise ValueError(f"AkShare 返回缺少必要列：{', '.join(missing)}")
    result = renamed.copy()
    result["stock_code"] = normalize_symbol(symbol)
    for column in ("volume", "amount"):
        if column not in result.columns:
            result[column] = pd.NA
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    return result[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return frame.loc[frame["date"].between(start_ts, end_ts)].reset_index(drop=True)


def _is_mainland_index(symbol: str) -> bool:
    code, _, suffix = normalize_symbol(symbol).partition(".")
    return (suffix == "SH" and code.startswith("000")) or (suffix == "SZ" and code.startswith("399"))


def _akshare_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")
