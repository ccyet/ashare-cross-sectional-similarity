from __future__ import annotations

import pandas as pd
import pytest

from scripts.download_all_a_daily import (
    _batched,
    _download_universe,
    _fetch_stock_symbols_for_engine,
    _group_symbols_by_download_start,
    fetch_all_a_symbols,
    merge_download_check,
)


def test_fetch_all_a_symbols_normalizes_akshare_code_column() -> None:
    symbols = fetch_all_a_symbols(
        fetcher=lambda: pd.DataFrame({"code": ["000001", "600519", "688603", "830799"]})
    )

    assert symbols == ["000001.SZ", "600519.SH", "688603.SH", "830799.BJ"]


def test_download_universe_adds_analysis_indexes_and_extra_symbols() -> None:
    symbols = _download_universe(
        ["000001.SZ", "600519.SH"],
        include_indexes=True,
        extra_symbols="399006, 000300.SH",
    )

    assert symbols[:2] == ["000001.SZ", "600519.SH"]
    assert "399006.SZ" in symbols
    assert "000300.SH" in symbols
    assert "000852.SH" in symbols


def test_fetch_stock_symbols_for_tdx_engine_uses_tdx(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_akshare() -> list[str]:
        raise AssertionError("tdx 模式不应调用 AkShare 股票列表")

    calls: list[str] = []

    def fake_tdx(*, tqcenter_path: str = "") -> list[str]:
        calls.append(tqcenter_path)
        return ["000001.SZ", "600519.SH"]

    monkeypatch.setattr("scripts.download_all_a_daily.fetch_all_a_symbols", fail_akshare)
    monkeypatch.setattr("scripts.download_all_a_daily.fetch_tdx_stock_symbols", fake_tdx)

    symbols = _fetch_stock_symbols_for_engine("tdx", "/tdx/PYPlugins/user")

    assert calls == ["/tdx/PYPlugins/user"]
    assert symbols == ["000001.SZ", "600519.SH"]


def test_merge_download_check_marks_missing_after_successful_delegate() -> None:
    download = pd.DataFrame(
        [{"symbol": "000001.SZ", "status": "delegated", "message": "ok"}]
    )
    checked = pd.DataFrame(
        [{"symbol": "000001.SZ", "status": "missing_file", "rows": 0, "message": "本地 parquet 不存在"}]
    )

    merged = merge_download_check(download, checked)

    assert merged["status"].tolist() == ["missing_file"]
    assert merged["message"].tolist() == ["下载后仍未覆盖：本地 parquet 不存在"]


def test_merge_download_check_keeps_download_failure_message() -> None:
    download = pd.DataFrame(
        [{"symbol": "000001.SZ", "status": "failed", "message": "provider failed"}]
    )
    checked = pd.DataFrame(
        [{"symbol": "000001.SZ", "status": "missing_file", "rows": 0, "message": "本地 parquet 不存在"}]
    )

    merged = merge_download_check(download, checked)

    assert merged["status"].tolist() == ["failed"]
    assert merged["message"].tolist() == ["provider failed"]


def test_batched_requires_positive_batch_size() -> None:
    assert list(_batched(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]


def test_group_symbols_by_download_start_keeps_incremental_batches_together() -> None:
    groups = _group_symbols_by_download_start(
        ["000001.SZ", "000002.SZ", "600519.SH"],
        {"000001.SZ": "2026-05-16", "600519.SH": "2026-05-20"},
        default_start="1990-01-01",
    )

    assert groups == [
        ("2026-05-16", ["000001.SZ"]),
        ("1990-01-01", ["000002.SZ"]),
        ("2026-05-20", ["600519.SH"]),
    ]
