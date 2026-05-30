from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_cross_section_similarity.cli import _parse_args
from ashare_cross_section_similarity.cli import _resolve_download_symbols
from ashare_cross_section_similarity.cli import _run_review
from ashare_cross_section_similarity.review import ReviewResult


def test_parse_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--symbols",
            "000001.SZ,600519.SH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--trend-repo",
            "/tmp/trend-backtest",
            "--provider",
            "tdx",
        ]
    )

    assert args.command == "download"
    assert args.symbols == "000001.SZ,600519.SH"
    assert args.timeframe == "1d"
    assert args.download_engine == "trend"
    assert args.trend_repo == "/tmp/trend-backtest"
    assert args.provider == "tdx"


def test_parse_search_date_tolerance_defaults_to_zero() -> None:
    args = _parse_args(
        [
            "search",
            "--target-symbol",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert args.command == "search"
    assert args.date_tolerance_bars == 0


def test_parse_search_date_tolerance_argument() -> None:
    args = _parse_args(
        [
            "search",
            "--target-symbol",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--date-tolerance-bars",
            "5",
        ]
    )

    assert args.date_tolerance_bars == 5


def test_parse_openbb_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--download-engine",
            "openbb",
            "--data-root",
            "/tmp/market/daily",
            "--symbols",
            "600519.SH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--provider",
            "akshare",
        ]
    )

    assert args.command == "download"
    assert args.download_engine == "openbb"
    assert args.data_root == "/tmp/market/daily"
    assert args.provider == "akshare"


def test_parse_native_akshare_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--download-engine",
            "akshare",
            "--data-root",
            "/tmp/market/daily",
            "--symbols",
            "600519.SH",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert args.command == "download"
    assert args.download_engine == "akshare"
    assert args.data_root == "/tmp/market/daily"


def test_parse_tdx_download_command() -> None:
    args = _parse_args(
        [
            "download",
            "--download-engine",
            "tdx",
            "--data-root",
            "/tmp/market/daily",
            "--symbols",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--provider",
            "/Applications/Tdx/PYPlugins/user",
        ]
    )

    assert args.command == "download"
    assert args.download_engine == "tdx"
    assert args.provider == "/Applications/Tdx/PYPlugins/user"


def test_parse_check_command() -> None:
    args = _parse_args(
        [
            "check",
            "--symbols",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert args.command == "check"
    assert args.symbols == "000001.SZ"


def test_parse_import_data_command() -> None:
    args = _parse_args(
        [
            "import-data",
            "--input",
            "/tmp/prices.csv",
            "--data-root",
            "/tmp/market/daily",
            "--fallback-symbol",
            "000001.SZ",
        ]
    )

    assert args.command == "import-data"
    assert args.input == "/tmp/prices.csv"
    assert args.data_root == "/tmp/market/daily"
    assert args.fallback_symbol == "000001.SZ"


def test_parse_search_command() -> None:
    args = _parse_args(
        [
            "search",
            "--target-symbol",
            "300750.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--universe-symbols",
            "000001.SZ",
        ]
    )

    assert args.command == "search"
    assert args.target_symbol == "300750.SZ"


def test_parse_history_command() -> None:
    args = _parse_args(
        [
            "history",
            "--symbol",
            "300750.SZ",
            "--as-of",
            "2024-03-31",
            "--window-size",
            "20",
        ]
    )

    assert args.command == "history"
    assert args.symbol == "300750.SZ"
    assert args.window_size == 20


def test_parse_review_command() -> None:
    args = _parse_args(
        [
            "review",
            "--target-symbol",
            "300750.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-31",
            "--model",
            "deepseek-v4-flash",
            "--output",
            "outputs/review.json",
        ]
    )

    assert args.command == "review"
    assert args.target_symbol == "300750.SZ"
    assert args.model == "deepseek-v4-flash"
    assert args.output == "outputs/review.json"


def test_review_evidence_only_writes_evidence_without_deepseek(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_path = tmp_path / "review.json"
    args = _parse_args(
        [
            "review",
            "--target-symbol",
            "300750.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--evidence-only",
            "--output",
            str(output_path),
        ]
    )
    review_result = ReviewResult(
        symbol="300750.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
        window=pd.DataFrame(),
        overview={},
        segments=pd.DataFrame(),
        main_segments=pd.DataFrame(),
        warnings=("缺少行情",),
    )

    monkeypatch.setattr("ashare_cross_section_similarity.cli.load_local_bars", lambda **_: pd.DataFrame())
    monkeypatch.setattr("ashare_cross_section_similarity.cli.analyze_price_review", lambda *_: review_result)
    monkeypatch.setattr(
        "ashare_cross_section_similarity.cli.DeepSeekClient",
        lambda *_: pytest.fail("evidence-only 不应调用 DeepSeek"),
    )

    assert _run_review(args) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["evidence"]["target"]["symbol"] == "300750.SZ"
    assert payload["evidence"]["warnings"] == ["缺少行情"]
    assert "ai_review" not in payload


def test_review_empty_window_stops_before_deepseek(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _parse_args(
        [
            "review",
            "--target-symbol",
            "300750.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--output",
            str(tmp_path / "review.json"),
        ]
    )
    review_result = ReviewResult(
        symbol="300750.SZ",
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-31"),
        window=pd.DataFrame(),
        overview={},
        segments=pd.DataFrame(),
        main_segments=pd.DataFrame(),
        warnings=("300750.SZ 在所选区间没有本地行情。",),
    )

    monkeypatch.setattr("ashare_cross_section_similarity.cli.load_local_bars", lambda **_: pd.DataFrame())
    monkeypatch.setattr("ashare_cross_section_similarity.cli.analyze_price_review", lambda *_: review_result)
    monkeypatch.setattr(
        "ashare_cross_section_similarity.cli.DeepSeekClient",
        lambda *_: pytest.fail("缺少本地行情时不应调用 DeepSeek"),
    )

    with pytest.raises(SystemExit, match="没有本地行情，未调用 DeepSeek"):
        _run_review(args)


def test_download_symbols_do_not_fallback_to_full_local_universe(tmp_path: Path) -> None:
    qfq = tmp_path / "market" / "daily" / "qfq"
    qfq.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "stock_code": ["600519.SH"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [1],
            "amount": [1],
        }
    ).to_parquet(qfq / "600519.SH.parquet", index=False)
    args = _parse_args(
        [
            "check",
            "--data-root",
            str(tmp_path / "market" / "daily"),
            "--symbols",
            "000001.SZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
        ]
    )

    assert _resolve_download_symbols(args) == ["000001.SZ"]
