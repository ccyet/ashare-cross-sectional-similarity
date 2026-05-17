from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from ashare_cross_section_similarity.data import (
    CANONICAL_COLUMNS,
    inclusive_end_timestamp,
    resolve_timeframe_root,
)
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

MINUTE_PERIODS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30"}
DATE_COLUMNS = ("date", "日期", "时间")
COLUMN_ALIASES = {
    "open": ("open", "开盘"),
    "high": ("high", "最高"),
    "low": ("low", "最低"),
    "close": ("close", "收盘"),
    "volume": ("volume", "成交量"),
    "amount": ("amount", "成交额"),
}


@dataclass(frozen=True)
class DownloadRequest:
    symbol: str
    timeframe: str
    start: str
    end: str
    adjust: str = "qfq"


class BarsClient(Protocol):
    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        ...


class AkshareBarsClient:
    def fetch(self, request: DownloadRequest) -> pd.DataFrame:
        import akshare as ak

        symbol = normalize_symbol(request.symbol)
        code = symbol.split(".", 1)[0]
        if request.timeframe == "1d":
            return self._fetch_daily(ak, symbol, code, request)
        if request.timeframe not in MINUTE_PERIODS:
            raise ValueError("timeframe 仅支持 1d、30m、15m、5m、1m。")
        return self._fetch_minute(ak, symbol, code, request)

    def _fetch_daily(self, ak, symbol: str, code: str, request: DownloadRequest) -> pd.DataFrame:
        start = pd.Timestamp(request.start).strftime("%Y%m%d")
        end = pd.Timestamp(request.end).strftime("%Y%m%d")
        asset_type = infer_asset_type(symbol)
        if asset_type == "index":
            return ak.index_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end)
        if asset_type == "etf":
            return ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=request.adjust,
            )
        return ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=request.adjust,
        )

    def _fetch_minute(self, ak, symbol: str, code: str, request: DownloadRequest) -> pd.DataFrame:
        start = _minute_api_timestamp(request.start, is_end=False)
        end = _minute_api_timestamp(request.end, is_end=True)
        period = MINUTE_PERIODS[request.timeframe]
        asset_type = infer_asset_type(symbol)
        if asset_type == "index":
            raise ValueError("AkShare 指数分钟线覆盖不稳定，请改用指数 ETF 或本地 parquet。")
        if asset_type == "etf":
            return ak.fund_etf_hist_min_em(
                symbol=code,
                period=period,
                start_date=start,
                end_date=end,
                adjust=request.adjust,
            )
        return ak.stock_zh_a_hist_min_em(
            symbol=code,
            period=period,
            start_date=start,
            end_date=end,
            adjust=request.adjust,
        )


def infer_asset_type(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    code, _, exchange = normalized.partition(".")
    if code.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "159", "16")):
        return "etf"
    if (exchange == "SH" and code.startswith(("000", "880", "881", "882"))) or (
        exchange == "SZ" and code.startswith(("399", "980"))
    ):
        return "index"
    return "stock"


def normalize_akshare_bars(
    raw: pd.DataFrame,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    frame = raw.copy()
    date_column = _find_column(frame, DATE_COLUMNS)
    if date_column is None:
        raise ValueError("行情数据缺少日期/时间列。")
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(frame[date_column], errors="coerce")
    result["stock_code"] = normalize_symbol(symbol)
    for canonical, aliases in COLUMN_ALIASES.items():
        source = _find_column(frame, aliases)
        result[canonical] = pd.to_numeric(frame[source], errors="coerce") if source else pd.NA
    start_ts = pd.Timestamp(start)
    end_ts = inclusive_end_timestamp(end)
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    result = result.loc[result["date"].between(start_ts, end_ts)]
    result = result[CANONICAL_COLUMNS].drop_duplicates(subset=["date"], keep="last")
    return result.sort_values("date").reset_index(drop=True)


def update_local_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    client: BarsClient | None = None,
) -> pd.DataFrame:
    client = client or AkshareBarsClient()
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for symbol in unique_symbols(symbols):
        request = DownloadRequest(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjust=adjust,
        )
        file_path = root / f"{symbol}.parquet"
        try:
            raw = client.fetch(request)
            downloaded = normalize_akshare_bars(raw, symbol, start, end)
            if downloaded.empty:
                raise ValueError("接口返回空行情。")
            merged = _merge_with_existing(file_path, downloaded)
            merged.to_parquet(file_path, index=False)
            rows.append(
                {
                    "symbol": symbol,
                    "status": "success",
                    "rows": int(len(merged)),
                    "new_rows": int(len(downloaded)),
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "symbol": symbol,
                    "status": "failed",
                    "rows": 0,
                    "new_rows": 0,
                    "message": str(exc),
                }
            )
    return pd.DataFrame(rows)


def data_check(
    *,
    symbols: tuple[str, ...] | list[str],
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    start_ts = pd.Timestamp(start)
    end_ts = inclusive_end_timestamp(end)
    rows: list[dict[str, object]] = []
    for symbol in unique_symbols(symbols):
        file_path = root / f"{symbol}.parquet"
        if not file_path.exists():
            rows.append(_check_row(symbol, "missing_file", 0, None, None, "本地 parquet 不存在"))
            continue
        try:
            frame = pd.read_parquet(file_path)
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            window = frame.loc[frame["date"].between(start_ts, end_ts)]
            status = "available" if not window.empty else "missing_window"
            message = "" if status == "available" else "所选区间无行情"
            rows.append(
                _check_row(
                    symbol,
                    status,
                    int(len(window)),
                    window["date"].min() if not window.empty else frame["date"].min(),
                    window["date"].max() if not window.empty else frame["date"].max(),
                    message,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(_check_row(symbol, "read_error", 0, None, None, str(exc)))
    return pd.DataFrame(rows)


def _merge_with_existing(file_path: Path, downloaded: pd.DataFrame) -> pd.DataFrame:
    if file_path.exists():
        existing = pd.read_parquet(file_path)
        combined = pd.concat([existing, downloaded], ignore_index=True)
    else:
        combined = downloaded
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return combined[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)


def _check_row(
    symbol: str,
    status: str,
    rows: int,
    start: object,
    end: object,
    message: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "rows": rows,
        "start": start,
        "end": end,
        "message": message,
    }


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        column = by_lower.get(candidate.lower())
        if column is not None:
            return column
    return None


def _minute_api_timestamp(value: str, *, is_end: bool) -> str:
    timestamp = pd.Timestamp(value)
    if " " not in value.strip() and "T" not in value.strip():
        timestamp = timestamp.replace(hour=15 if is_end else 9, minute=0 if is_end else 30)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")
