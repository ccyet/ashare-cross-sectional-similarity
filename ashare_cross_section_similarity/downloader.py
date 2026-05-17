from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable

import pandas as pd

from ashare_cross_section_similarity.data import inclusive_end_timestamp, resolve_timeframe_root
from ashare_cross_section_similarity.universe import unique_symbols

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
    provider: str = "",
    runner: CommandRunner | None = None,
) -> pd.DataFrame:
    normalized_symbols = unique_symbols(symbols)
    if not normalized_symbols:
        return pd.DataFrame(columns=["symbol", "status", "rows", "new_rows", "message"])

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
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "status": status,
        "rows": rows,
        "start": start,
        "end": end,
        "message": message,
    }
