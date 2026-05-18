from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable

import pandas as pd

from ashare_cross_section_similarity.data import (
    CANONICAL_COLUMNS,
    inclusive_end_timestamp,
    resolve_timeframe_root,
)
from ashare_cross_section_similarity.openbb_source import fetch_openbb_bars
from ashare_cross_section_similarity.tdx_source import fetch_tdx_bars
from ashare_cross_section_similarity.universe import normalize_symbol, unique_symbols

CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


def default_trend_repo() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "trend-backtest",
        Path("/Users/a1234/Desktop/trend-backtest"),
    ]
    for candidate in candidates:
        if (candidate / "scripts" / "update_data.py").exists():
            return candidate
    return candidates[-1]


def build_update_command(
    *,
    trend_repo: str | Path,
    symbols: tuple[str, ...] | list[str],
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    provider: str = "",
) -> list[str]:
    repo = Path(trend_repo).expanduser()
    script = repo / "scripts" / "update_data.py"
    if not script.exists():
        raise FileNotFoundError(f"未找到原库数据更新脚本：{script}")

    command = [
        sys.executable,
        str(script),
        "--symbols",
        ",".join(unique_symbols(symbols)),
        "--start-date",
        start,
        "--end-date",
        end,
        "--timeframe",
        timeframe,
    ]
    if adjust:
        command.extend(["--adjust", adjust])
    if provider:
        command.extend(["--provider", f"{timeframe}={provider}"])
    return command


def update_local_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    trend_repo: str | Path | None = None,
    data_root: str | Path = "data/market/daily",
    provider: str = "",
    download_engine: str = "trend",
    runner: CommandRunner | None = None,
) -> pd.DataFrame:
    normalized_symbols = unique_symbols(symbols)
    if not normalized_symbols:
        return pd.DataFrame(columns=["symbol", "status", "rows", "new_rows", "message"])
    if download_engine not in {"trend", "openbb", "tdx"}:
        raise ValueError("download_engine 仅支持 trend、openbb 或 tdx。")
    if download_engine == "openbb":
        return _update_local_bars_with_openbb(
            symbols=normalized_symbols,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
            data_root=data_root,
            provider=provider or "akshare",
        )
    if download_engine == "tdx":
        return _update_local_bars_with_tdx(
            symbols=normalized_symbols,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
            data_root=data_root,
            tqcenter_path=provider,
        )

    repo = Path(trend_repo).expanduser() if trend_repo else default_trend_repo()
    command = build_update_command(
        trend_repo=repo,
        symbols=normalized_symbols,
        timeframe=timeframe,
        adjust=adjust,
        start=start,
        end=end,
        provider=provider,
    )
    runner = runner or _run_command
    completed = runner(command, repo)
    if completed.returncode == 0:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "status": "delegated",
                    "rows": 0,
                    "new_rows": 0,
                    "message": "原 update_data.py 已执行；请用 check 查看落地覆盖。",
                }
                for symbol in normalized_symbols
            ]
        )

    message = (completed.stderr or completed.stdout or "原 update_data.py 执行失败").strip()
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "status": "failed",
                "rows": 0,
                "new_rows": 0,
                "message": message,
            }
            for symbol in normalized_symbols
        ]
    )


def _update_local_bars_with_openbb(
    *,
    symbols: list[str],
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    data_root: str | Path,
    provider: str,
) -> pd.DataFrame:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            frame = fetch_openbb_bars(
                symbols=(symbol,),
                start=start,
                end=end,
                provider=provider,
                timeframe=timeframe,
            )
            frame = frame.loc[frame["stock_code"] == symbol, CANONICAL_COLUMNS]
            if frame.empty:
                rows.append(_download_row(symbol, "failed", 0, 0, "OpenBB 未返回行情数据"))
                continue
            saved = _write_symbol_bars(root / f"{symbol}.parquet", frame)
            rows.append(
                _download_row(
                    symbol,
                    "success",
                    int(len(saved)),
                    int(len(frame)),
                    f"OpenBB/{provider} 行情已写入本地 parquet。",
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(_download_row(symbol, "failed", 0, 0, str(exc)))
    return pd.DataFrame(rows)


def _update_local_bars_with_tdx(
    *,
    symbols: list[str],
    timeframe: str,
    adjust: str,
    start: str,
    end: str,
    data_root: str | Path,
    tqcenter_path: str,
) -> pd.DataFrame:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    try:
        fetched = fetch_tdx_bars(
            symbols=tuple(symbols),
            start=start,
            end=end,
            timeframe=timeframe,
            adjust=adjust,
            tqcenter_path=tqcenter_path,
        )
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame([_download_row(symbol, "failed", 0, 0, str(exc)) for symbol in symbols])

    for symbol in symbols:
        try:
            frame = fetched.loc[fetched["stock_code"] == symbol, CANONICAL_COLUMNS]
            if frame.empty:
                rows.append(_download_row(symbol, "failed", 0, 0, "TDX 未返回行情数据"))
                continue
            saved = _write_symbol_bars(root / f"{symbol}.parquet", frame)
            rows.append(
                _download_row(
                    symbol,
                    "success",
                    int(len(saved)),
                    int(len(frame)),
                    "TDX 行情已写入本地 parquet。",
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(_download_row(symbol, "failed", 0, 0, str(exc)))
    return pd.DataFrame(rows)


def _write_symbol_bars(file_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if file_path.exists():
        frames.append(_normalize_download_frame(pd.read_parquet(file_path)))
    frames.append(_normalize_download_frame(frame))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[CANONICAL_COLUMNS].drop_duplicates(subset=["stock_code", "date"], keep="last")
    merged = merged.sort_values(["stock_code", "date"]).reset_index(drop=True)
    merged.to_parquet(file_path, index=False)
    return merged


def _normalize_download_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "stock_code" not in result.columns and "symbol" in result.columns:
        result = result.rename(columns={"symbol": "stock_code"})
    if "stock_code" not in result.columns:
        raise ValueError("行情数据缺少 stock_code 或 symbol 列。")
    result["stock_code"] = result["stock_code"].map(normalize_symbol)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "stock_code", "open", "high", "low", "close"])
    return result[CANONICAL_COLUMNS]


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
    requested_start_day = start_ts.normalize()
    requested_end_day = pd.Timestamp(end).normalize()
    rows: list[dict[str, object]] = []
    for symbol in unique_symbols(symbols):
        file_path = root / f"{symbol}.parquet"
        if not file_path.exists():
            rows.append(
                _check_row(
                    symbol,
                    "missing_file",
                    0,
                    None,
                    None,
                    "本地 parquet 不存在",
                    requested_start=requested_start_day,
                    requested_end=requested_end_day,
                )
            )
            continue
        try:
            frame = pd.read_parquet(file_path, columns=["date"])
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            window = frame.loc[frame["date"].between(start_ts, end_ts)]
            local_start = frame["date"].min()
            local_end = frame["date"].max()
            if window.empty:
                status = "missing_window"
                message = "所选区间无行情"
            else:
                window_start_day = pd.Timestamp(window["date"].min()).normalize()
                window_end_day = pd.Timestamp(window["date"].max()).normalize()
                status = "available"
                message = ""
                if window_start_day > requested_start_day or window_end_day < requested_end_day:
                    status = "partial_window"
                    message = f"区间覆盖不足，实际覆盖 {_date_text(window_start_day)} 至 {_date_text(window_end_day)}"
            rows.append(
                _check_row(
                    symbol,
                    status,
                    int(len(window)),
                    window["date"].min() if not window.empty else frame["date"].min(),
                    window["date"].max() if not window.empty else frame["date"].max(),
                    message,
                    requested_start=requested_start_day,
                    requested_end=requested_end_day,
                    local_start=local_start,
                    local_end=local_end,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _check_row(
                    symbol,
                    "read_error",
                    0,
                    None,
                    None,
                    str(exc),
                    requested_start=requested_start_day,
                    requested_end=requested_end_day,
                )
            )
    return pd.DataFrame(rows)


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _check_row(
    symbol: str,
    status: str,
    rows: int,
    start: object,
    end: object,
    message: str,
    *,
    requested_start: object = None,
    requested_end: object = None,
    local_start: object = None,
    local_end: object = None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "rows": rows,
        "start": start,
        "end": end,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "local_start": local_start if local_start is not None else start,
        "local_end": local_end if local_end is not None else end,
        "message": message,
    }


def _download_row(
    symbol: str,
    status: str,
    rows: int,
    new_rows: int,
    message: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "rows": rows,
        "new_rows": new_rows,
        "message": message,
    }
