from __future__ import annotations

from io import BufferedIOBase
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import pyarrow.parquet as pq

from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

TIMEFRAME_DIR_NAMES = {
    "1d": "daily",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}

CANONICAL_COLUMNS = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]
IMPORT_STATUS_COLUMNS = ["symbol", "status", "rows", "total_rows", "start", "end", "message"]


def resolve_timeframe_root(data_root: str | Path, timeframe: str) -> Path:
    root = Path(data_root).expanduser()
    timeframe_dir = TIMEFRAME_DIR_NAMES.get(str(timeframe), str(timeframe))
    if timeframe == "1d":
        return root
    if root.name.lower() == "daily":
        return root.parent / timeframe_dir
    return root / timeframe_dir if root.name.lower() not in TIMEFRAME_DIR_NAMES.values() else root


def available_symbols(data_root: str | Path, timeframe: str = "1d", adjust: str = "qfq") -> list[str]:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    if not root.exists():
        return []
    return sorted(normalize_symbol(path.stem) for path in root.glob("*.parquet"))


def load_local_bars(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    start_ts = pd.Timestamp(start)
    end_ts = inclusive_end_timestamp(end)
    frames: list[pd.DataFrame] = []
    for symbol in unique_symbols(symbols):
        file_path = root / f"{symbol}.parquet"
        if not file_path.exists():
            continue
        frame = _normalize_bars(_read_bars_parquet(file_path), symbol)
        frame = frame.loc[frame["date"].between(start_ts, end_ts)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def read_price_data_file(file: str | Path | BinaryIO | BufferedIOBase, filename: str | None = None) -> pd.DataFrame:
    file_name = filename or str(file)
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file)
    if suffix == ".parquet":
        return pd.read_parquet(file)
    raise ValueError("价格数据文件仅支持 csv 或 parquet。")


def import_price_frame(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    frame: pd.DataFrame,
    fallback_symbol: str = "",
    source_name: str = "",
) -> pd.DataFrame:
    fallback = normalize_symbol(fallback_symbol) if fallback_symbol else ""
    _validate_price_frame_columns(frame, fallback)
    imported = _normalize_bars(frame, fallback)
    imported = imported.loc[imported["stock_code"] != ""]
    if imported.empty:
        raise ValueError("价格数据没有可导入的有效 OHLC 行。")

    root = resolve_timeframe_root(data_root, timeframe) / adjust
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for symbol, symbol_frame in imported.groupby("stock_code", sort=True):
        file_path = root / f"{symbol}.parquet"
        if file_path.exists():
            existing = _normalize_bars(_read_bars_parquet(file_path), symbol)
            merged = pd.concat([existing, symbol_frame], ignore_index=True)
        else:
            merged = symbol_frame.copy()
        merged = (
            merged[CANONICAL_COLUMNS]
            .drop_duplicates(subset=["stock_code", "date"], keep="last")
            .sort_values(["stock_code", "date"])
            .reset_index(drop=True)
        )
        merged.to_parquet(file_path, index=False)
        rows.append(
            {
                "symbol": symbol,
                "status": "imported",
                "rows": int(len(symbol_frame)),
                "total_rows": int(len(merged)),
                "start": merged["date"].min(),
                "end": merged["date"].max(),
                "message": f"已导入 {source_name or '价格数据'}",
            }
        )
    return pd.DataFrame(rows, columns=IMPORT_STATUS_COLUMNS)


def inclusive_end_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if isinstance(value, str):
        text = value.strip()
        if " " not in text and "T" not in text and len(text) <= 10:
            return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def _normalize_bars(frame: pd.DataFrame, fallback_symbol: str) -> pd.DataFrame:
    result = frame.copy()
    if "stock_code" not in result.columns and "symbol" in result.columns:
        result = result.rename(columns={"symbol": "stock_code"})
    if "stock_code" not in result.columns:
        result["stock_code"] = fallback_symbol
    result["stock_code"] = result["stock_code"].map(normalize_symbol)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    result = result[CANONICAL_COLUMNS].drop_duplicates(subset=["stock_code", "date"], keep="last")
    return result.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _validate_price_frame_columns(frame: pd.DataFrame, fallback_symbol: str) -> None:
    missing = [column for column in ["date", "open", "high", "low", "close"] if column not in frame.columns]
    if missing:
        raise ValueError(f"价格数据缺少必要列：{', '.join(missing)}。")
    if "stock_code" not in frame.columns and "symbol" not in frame.columns and not fallback_symbol:
        raise ValueError("价格数据必须包含 symbol/stock_code 列，或提供 fallback_symbol。")


def _read_bars_parquet(file_path: Path) -> pd.DataFrame:
    available_columns = set(pq.read_schema(file_path).names)
    columns = [
        column
        for column in CANONICAL_COLUMNS + ["symbol"]
        if column in available_columns
    ]
    return pd.read_parquet(file_path, columns=columns)
