from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ashare_cross_section_similarity.similarity as similarity_module
from ashare_cross_section_similarity.features import FEATURE_COLUMNS, window_features
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionWindowTraversalConfig,
    search_cross_section,
    search_cross_section_window_traversal,
)


def _bars(
    symbol: str,
    closes: list[float],
    amounts: list[float] | None = None,
    *,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(closes), freq="D")
    amounts = amounts or [1000.0 + index * 10 for index in range(len(closes))]
    return pd.DataFrame(
        {
            "date": dates,
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [100.0 + index for index in range(len(closes))],
            "amount": amounts,
        }
    )


def test_search_ranks_same_time_universe_by_target_interval_shape() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 11, 13]),
            _bars("000002.SZ", [20, 22, 24, 22, 26]),
            _bars("000003.SZ", [10, 9, 8, 7, 6]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ", "000003.SZ"),
            start="2024-01-01",
            end="2024-01-05",
            top_n=2,
        ),
    )

    assert result.target_symbol == "000001.SZ"
    assert result.window_size == 5
    assert result.results["symbol"].tolist() == ["000002.SZ", "000003.SZ"]
    assert result.results["综合相似度"].iloc[0] > result.results["综合相似度"].iloc[1]
    assert result.results["路径相似度"].between(0, 1).all()


def test_search_excludes_target_symbol_and_reports_skip_reasons() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13, 14]),
            _bars("000002.SZ", [10, 11]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000001.SZ", "000002.SZ"),
            start="2024-01-01",
            end="2024-01-05",
            min_coverage=0.8,
        ),
    )

    assert result.results.empty
    assert result.skipped["symbol"].tolist() == ["000002.SZ"]
    assert "区间数据不足" in result.skipped["原因"].iloc[0]


def test_search_uses_same_window_semantics_after_grouped_filtering() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [99, 10, 11, 12, 11, 13, 98]),
            _bars("000002.SZ", [88, 20, 22, 24, 22, 26, 87]),
            _bars("000003.SZ", [77, 10, 9, 8, 7, 6, 76]),
            _bars("000004.SZ", [66, 10, 11, 65, 64, 63, 62]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ", "000003.SZ", "000004.SZ"),
            start="2024-01-02",
            end="2024-01-06",
            top_n=3,
            min_coverage=1.0,
        ),
    )

    assert result.window_size == 5
    assert result.results["symbol"].tolist() == ["000002.SZ", "000004.SZ", "000003.SZ"]
    assert result.results["区间开始"].tolist() == [pd.Timestamp("2024-01-02")] * 3
    assert result.results["区间结束"].tolist() == [pd.Timestamp("2024-01-06")] * 3


def test_search_date_tolerance_zero_keeps_strict_same_day_semantics() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [99, 10, 11, 12, 11, 13, 98]),
            _bars("000002.SZ", [88, 20, 22, 24, 22, 26, 87]),
            _bars("000003.SZ", [77, 10, 9, 8, 7, 6, 76]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ", "000003.SZ"),
            start="2024-01-02",
            end="2024-01-06",
            top_n=2,
            min_coverage=1.0,
            date_tolerance_bars=0,
        ),
    )

    assert result.window_size == 5
    assert result.results["symbol"].tolist() == ["000002.SZ", "000003.SZ"]
    assert result.results["区间开始"].tolist() == [pd.Timestamp("2024-01-02")] * 2
    assert result.results["区间结束"].tolist() == [pd.Timestamp("2024-01-06")] * 2
    assert result.results["日期偏移"].tolist() == [0, 0]
    assert result.results["覆盖率"].tolist() == [1.0, 1.0]


def test_search_date_tolerance_picks_shifted_candidate_window() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [6, 7, 8, 10, 12, 11, 13, 14, 15, 16, 17, 18]),
            _bars("000002.SZ", [1, 1, 1, 2, 2, 2, 10, 12, 11, 22, 33, 44]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ",),
            start="2024-01-04",
            end="2024-01-06",
            top_n=1,
            min_coverage=1.0,
            date_tolerance_bars=3,
            forward_windows=(3,),
        ),
    )

    row = result.results.iloc[0]
    assert row["区间开始"] == pd.Timestamp("2024-01-07")
    assert row["区间结束"] == pd.Timestamp("2024-01-09")
    assert row["日期偏移"] == 3
    assert row["覆盖率"] == 1.0
    assert row["t_plus_3_return"] == pytest.approx(44 / 11 - 1)


def test_search_date_tolerance_scores_offsets_without_per_offset_z_normalize(monkeypatch) -> None:
    symbols = [f"{index:06d}.SZ" for index in range(1, 8)]
    bars = pd.concat(
        [
            _bars(
                symbol,
                (np.linspace(10, 20, 30) * (1 + index * 0.01)).tolist(),
            )
            for index, symbol in enumerate(symbols)
        ],
        ignore_index=True,
    )
    call_count = 0
    original_z_normalize = similarity_module.z_normalize

    def counted_z_normalize(values: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original_z_normalize(values)

    monkeypatch.setattr(similarity_module, "z_normalize", counted_z_normalize)

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=tuple(symbols),
            start="2024-01-08",
            end="2024-01-17",
            top_n=3,
            min_coverage=1.0,
            date_tolerance_bars=5,
        ),
    )

    assert len(result.results) == 3
    assert call_count <= 2


def test_window_traversal_matches_target_window_inside_historical_range() -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 12, 11, 13], start="2026-05-17"),
            _bars("000001.SZ", [7, 8, 10, 12, 11, 13, 14], start="2021-01-01"),
            _bars("600519.SH", [30, 29, 28, 27, 26, 25, 24], start="2021-01-01"),
        ],
        ignore_index=True,
    )

    result = search_cross_section_window_traversal(
        bars,
        CrossSectionWindowTraversalConfig(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ", "600519.SH"),
            target_start="2026-05-17",
            target_end="2026-05-20",
            traversal_start="2021-01-01",
            traversal_end="2021-01-07",
            top_n=2,
            min_coverage=1.0,
            forward_windows=(1,),
        ),
    )

    row = result.results.iloc[0]
    assert result.window_size == 4
    assert row["symbol"] == "000001.SZ"
    assert row["区间开始"] == pd.Timestamp("2021-01-03")
    assert row["区间结束"] == pd.Timestamp("2021-01-06")
    assert row["遍历偏移"] == 2
    assert row["t_plus_1_return"] == pytest.approx(14 / 13 - 1)


def test_window_traversal_excludes_nearby_candidate_windows() -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 11, 12, 13], start="2026-05-17"),
            _bars("000001.SZ", list(range(10, 26)), start="2021-01-01"),
        ],
        ignore_index=True,
    )

    result = search_cross_section_window_traversal(
        bars,
        CrossSectionWindowTraversalConfig(
            target_symbol="300750.SZ",
            universe_symbols=("000001.SZ",),
            target_start="2026-05-17",
            target_end="2026-05-20",
            traversal_start="2021-01-01",
            traversal_end="2021-01-16",
            top_n=3,
            min_coverage=1.0,
            exclusion_bars=2,
        ),
    )

    starts = sorted(pd.to_datetime(result.results["区间开始"]).tolist())
    assert len(starts) == 3
    assert all((right - left).days > 2 for left, right in zip(starts, starts[1:]))


def test_window_traversal_scores_candidate_windows_with_vectorized_features(monkeypatch) -> None:
    bars = pd.concat(
        [
            _bars("300750.SZ", [10, 12, 11, 13], start="2026-05-17"),
            *[
                _bars(
                    f"{index:06d}.SZ",
                    (np.linspace(8 + index, 30 + index, 80) + np.sin(np.arange(80))).tolist(),
                    start="2021-01-01",
                )
                for index in range(1, 5)
            ],
        ],
        ignore_index=True,
    )
    call_count = 0
    original = similarity_module._fast_window_features

    def counted_fast_window_features(window: pd.DataFrame) -> dict[str, float]:
        nonlocal call_count
        call_count += 1
        return original(window)

    monkeypatch.setattr(similarity_module, "_fast_window_features", counted_fast_window_features)

    result = search_cross_section_window_traversal(
        bars,
        CrossSectionWindowTraversalConfig(
            target_symbol="300750.SZ",
            universe_symbols=tuple(f"{index:06d}.SZ" for index in range(1, 5)),
            target_start="2026-05-17",
            target_end="2026-05-20",
            traversal_start="2021-01-01",
            traversal_end="2021-03-21",
            top_n=5,
            min_coverage=1.0,
        ),
    )

    assert len(result.results) == 5
    assert call_count == 1


def test_fast_cross_section_features_match_public_window_features() -> None:
    bars = _bars("000001.SZ", [10, 11, 9, 12, 13], amounts=[1000, 1200, 1800, 1300, 1400])

    expected = window_features(bars)
    actual = similarity_module._fast_window_features(bars)

    for column in FEATURE_COLUMNS:
        assert actual[column] == pytest.approx(expected[column])


def test_search_reports_forward_returns_after_historical_window() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
            _bars("000002.SZ", [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ",),
            start="2024-01-01",
            end="2024-01-02",
            top_n=1,
            forward_windows=(3, 5, 10),
        ),
    )

    row = result.results.iloc[0]
    assert row["t_plus_3_return"] == pytest.approx(28 / 22 - 1)
    assert row["t_plus_5_return"] == pytest.approx(32 / 22 - 1)
    assert row["t_plus_10_return"] == pytest.approx(42 / 22 - 1)
