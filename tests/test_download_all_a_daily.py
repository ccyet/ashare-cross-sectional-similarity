from __future__ import annotations

import pandas as pd

from scripts.download_all_a_daily import _batched, _download_universe, fetch_all_a_symbols, merge_download_check


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
