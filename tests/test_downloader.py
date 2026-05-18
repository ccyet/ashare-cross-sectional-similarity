from __future__ import annotations

from pathlib import Path
import subprocess
from unittest.mock import patch

import pandas as pd

from ashare_cross_section_similarity.downloader import (
    build_update_command,
    data_check,
    update_local_bars,
)


def test_build_update_command_uses_original_trend_backtest_script(tmp_path: Path) -> None:
    repo = tmp_path / "trend-backtest"
    script = repo / "scripts" / "update_data.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder", encoding="utf-8")

    command = build_update_command(
        trend_repo=repo,
        symbols=("000001.SZ", "600519.SH"),
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        provider="akshare",
    )

    assert command[1] == str(script)
    assert "--symbols" in command
    assert "000001.SZ,600519.SH" in command
    assert ["--timeframe", "1d"] == command[command.index("--timeframe") : command.index("--timeframe") + 2]
    assert ["--provider", "1d=akshare"] == command[command.index("--provider") : command.index("--provider") + 2]


def test_update_local_bars_delegates_to_original_runner(tmp_path: Path) -> None:
    repo = tmp_path / "trend-backtest"
    script = repo / "scripts" / "update_data.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = update_local_bars(
        symbols=("000001.SZ",),
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=repo,
        runner=runner,
    )

    assert calls[0][1] == repo
    assert calls[0][0][1] == str(script)
    assert result["status"].tolist() == ["delegated"]
    assert "原 update_data.py 已执行" in result["message"].iloc[0]


def test_update_local_bars_reports_original_runner_failure(tmp_path: Path) -> None:
    repo = tmp_path / "trend-backtest"
    script = repo / "scripts" / "update_data.py"
    script.parent.mkdir(parents=True)
    script.write_text("# placeholder", encoding="utf-8")

    def runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="provider failed")

    result = update_local_bars(
        symbols=("000001.SZ",),
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-31",
        trend_repo=repo,
        runner=runner,
    )

    assert result["status"].tolist() == ["failed"]
    assert "provider failed" in result["message"].iloc[0]


def test_update_local_bars_can_fetch_and_write_with_openbb_engine(tmp_path: Path) -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "stock_code": ["600519.SH", "600519.SH"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
            "amount": [100, 200],
        }
    )

    with patch(
        "ashare_cross_section_similarity.downloader.fetch_openbb_bars",
        return_value=bars,
    ) as fetch_openbb_bars:
        result = update_local_bars(
            symbols=("600519.SH",),
            timeframe="1d",
            adjust="qfq",
            start="2024-01-01",
            end="2024-01-02",
            data_root=tmp_path / "market" / "daily",
            provider="akshare",
            download_engine="openbb",
        )

    fetch_openbb_bars.assert_called_once_with(
        symbols=("600519.SH",),
        start="2024-01-01",
        end="2024-01-02",
        provider="akshare",
        timeframe="1d",
    )
    assert result[["symbol", "status", "rows", "new_rows"]].to_dict("records") == [
        {"symbol": "600519.SH", "status": "success", "rows": 2, "new_rows": 2}
    ]
    saved = pd.read_parquet(tmp_path / "market" / "daily" / "qfq" / "600519.SH.parquet")
    assert saved["close"].tolist() == [1, 2]


def test_openbb_engine_merges_with_existing_parquet_using_canonical_schema(
    tmp_path: Path,
) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "symbol": ["600519.SH"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
        }
    ).to_parquet(qfq / "600519.SH.parquet", index=False)
    bars = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "stock_code": ["600519.SH"],
            "open": [2],
            "high": [2],
            "low": [2],
            "close": [2],
            "volume": [20],
            "amount": [200],
        }
    )

    with patch(
        "ashare_cross_section_similarity.downloader.fetch_openbb_bars",
        return_value=bars,
    ):
        result = update_local_bars(
            symbols=("600519.SH",),
            timeframe="1d",
            adjust="qfq",
            start="2024-01-02",
            end="2024-01-02",
            data_root=tmp_path / "market" / "daily",
            download_engine="openbb",
        )

    saved = pd.read_parquet(qfq / "600519.SH.parquet")
    assert result[["symbol", "status", "rows", "new_rows"]].to_dict("records") == [
        {"symbol": "600519.SH", "status": "success", "rows": 2, "new_rows": 1}
    ]
    assert saved.columns.tolist() == [
        "date",
        "stock_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert saved["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert saved["volume"].isna().iloc[0]
    assert saved["amount"].isna().iloc[0]


def test_data_check_reports_missing_and_available_symbols(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
            "amount": [100, 200],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = data_check(
        symbols=("000001.SZ", "000002.SZ"),
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-02",
    )

    assert out[["symbol", "status", "rows"]].to_dict("records") == [
        {"symbol": "000001.SZ", "status": "available", "rows": 2},
        {"symbol": "000002.SZ", "status": "missing_file", "rows": 0},
    ]


def test_data_check_reports_partial_window_when_range_is_not_fully_covered(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-08"],
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 20],
            "amount": [100, 200],
        }
    ).to_parquet(qfq / "000001.SZ.parquet", index=False)

    out = data_check(
        symbols=("000001.SZ",),
        data_root=tmp_path / "market" / "daily",
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-10",
    )

    row = out.iloc[0]
    assert row["status"] == "partial_window"
    assert row["rows"] == 2
    assert row["start"] == pd.Timestamp("2024-01-05")
    assert row["end"] == pd.Timestamp("2024-01-08")
    assert "覆盖不足" in row["message"]


def test_data_check_reads_only_date_column(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    file_path = qfq / "000001.SZ.parquet"
    file_path.write_bytes(b"placeholder")
    frame = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})

    with patch("pandas.read_parquet", return_value=frame) as read_parquet:
        out = data_check(
            symbols=("000001.SZ",),
            data_root=tmp_path / "market" / "daily",
            timeframe="1d",
            adjust="qfq",
            start="2024-01-01",
            end="2024-01-02",
        )

    read_parquet.assert_called_once_with(file_path, columns=["date"])
    assert out["status"].tolist() == ["available"]
