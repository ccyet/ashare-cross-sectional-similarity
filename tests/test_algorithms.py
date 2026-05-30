from __future__ import annotations

import pandas as pd
import pytest

from ashare_cross_section_similarity.history import HistorySearchConfig, search_history
from ashare_cross_section_similarity.similarity import CrossSectionSearchConfig, search_cross_section
from ashare_cross_section_similarity.similarity_algorithms import (
    available_algorithm_names,
    build_algorithm_target,
    distance_for_window,
    get_algorithm_status,
)


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": [100.0 + index for index in range(len(closes))],
            "amount": [1000.0 + index * 10 for index in range(len(closes))],
        }
    )


def test_algorithm_registry_reports_baseline_and_optional_status() -> None:
    names = available_algorithm_names()

    assert names[:3] == ["baseline_price_feature", "return_shape", "hybrid_shape_v2"]
    assert get_algorithm_status("baseline_price_feature").available is True
    assert get_algorithm_status("dtw_optional").name == "dtw_optional"
    assert get_algorithm_status("mass_optional_history").name == "mass_optional_history"


def test_cross_section_explicit_baseline_matches_default() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12, 11, 13]),
            _bars("000002.SZ", [20, 22, 24, 22, 26]),
            _bars("000003.SZ", [10, 9, 8, 7, 6]),
        ],
        ignore_index=True,
    )
    common = dict(
        target_symbol="000001.SZ",
        universe_symbols=("000002.SZ", "000003.SZ"),
        start="2024-01-01",
        end="2024-01-05",
        top_n=2,
    )

    default = search_cross_section(bars, CrossSectionSearchConfig(**common)).results
    explicit = search_cross_section(
        bars,
        CrossSectionSearchConfig(**common, algorithm="baseline_price_feature"),
    ).results

    assert explicit["算法"].tolist() == ["baseline_price_feature", "baseline_price_feature"]
    pd.testing.assert_series_equal(default["symbol"], explicit["symbol"])
    pd.testing.assert_series_equal(default["综合相似度"], explicit["综合相似度"])


def test_return_shape_cross_section_uses_return_path_for_ranking() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [100, 110, 99, 108.9, 98.01]),
            _bars("000002.SZ", [30, 33, 29.7, 32.67, 29.403]),
            _bars("000003.SZ", [100, 105, 110, 115, 120]),
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
            algorithm="return_shape",
            top_n=2,
        ),
    )

    assert result.results["symbol"].tolist()[0] == "000002.SZ"
    assert result.results["算法"].tolist()[0] == "return_shape"
    assert result.results["收益路径距离"].iloc[0] < result.results["收益路径距离"].iloc[1]


def test_hybrid_shape_v2_exposes_distance_components() -> None:
    bars = pd.concat(
        [
            _bars("000001.SZ", [100, 110, 99, 108.9, 98.01]),
            _bars("000002.SZ", [30, 33, 29.7, 32.67, 29.403]),
        ],
        ignore_index=True,
    )

    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol="000001.SZ",
            universe_symbols=("000002.SZ",),
            start="2024-01-01",
            end="2024-01-05",
            algorithm="hybrid_shape_v2",
        ),
    )

    row = result.results.iloc[0]
    assert row["算法"] == "hybrid_shape_v2"
    assert row["价格路径距离"] == pytest.approx(0.0)
    assert row["收益路径距离"] == pytest.approx(0.0)
    assert row["路径距离"] == pytest.approx(0.0)


def test_price_distance_emphasizes_recent_two_klines() -> None:
    target = build_algorithm_target(_bars("000001.SZ", [1, 2, 3, 4, 5]), "baseline_price_feature")
    early_mismatch = distance_for_window(_bars("000002.SZ", [1, 2.5, 3, 4, 5]), target)
    recent_mismatch = distance_for_window(_bars("000003.SZ", [1, 2, 3, 4, 5.5]), target)

    assert recent_mismatch["价格路径距离"] > early_mismatch["价格路径距离"]


def test_return_distance_emphasizes_recent_two_klines() -> None:
    target = build_algorithm_target(
        _bars("000001.SZ", [100.0, 103.1156, 100.8102, 102.3621, 102.6995, 103.7769]),
        "return_shape",
    )
    early_mismatch = distance_for_window(
        _bars("000002.SZ", [100.0, 103.1156, 91.8332, 102.3621, 102.6995, 103.7769]),
        target,
    )
    recent_mismatch = distance_for_window(
        _bars("000003.SZ", [100.0, 103.1156, 100.8102, 102.3621, 102.6995, 101.6373]),
        target,
    )

    assert recent_mismatch["收益路径距离"] > early_mismatch["收益路径距离"]


def test_single_axis_algorithms_skip_unused_distance_components() -> None:
    baseline_target = build_algorithm_target(_bars("000001.SZ", [1, 2, 3, 4, 5]), "baseline_price_feature")
    baseline = distance_for_window(_bars("000002.SZ", [1, 2, 3, 4, 5]), baseline_target)
    return_target = build_algorithm_target(_bars("000001.SZ", [1, 2, 3, 4, 5]), "return_shape")
    return_shape = distance_for_window(_bars("000002.SZ", [1, 2, 3, 4, 5]), return_target)

    assert pd.isna(baseline["收益路径距离"])
    assert pd.isna(return_shape["价格路径距离"])


def test_history_return_shape_ranks_matching_return_window_first() -> None:
    bars = _bars(
        "000001.SZ",
        [
            50, 55, 49.5, 54.45, 49.005,
            70, 72, 74, 76, 78,
            100, 110, 99, 108.9, 98.01,
        ],
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            algorithm="return_shape",
            forward_windows=(1,),
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    assert result.results["窗口开始"].iloc[0] == pd.Timestamp("2024-01-01")
    assert result.results["算法"].iloc[0] == "return_shape"
    assert result.results["收益路径距离"].iloc[0] == pytest.approx(0.0)


def test_optional_dtw_dependency_missing_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import ashare_cross_section_similarity.similarity_algorithms as algorithms

    monkeypatch.setattr(algorithms, "_optional_dtw_distance", lambda _left, _right: None)
    bars = pd.concat(
        [
            _bars("000001.SZ", [10, 11, 12]),
            _bars("000002.SZ", [10, 11, 12]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="dtw_optional"):
        search_cross_section(
            bars,
            CrossSectionSearchConfig(
                target_symbol="000001.SZ",
                universe_symbols=("000002.SZ",),
                start="2024-01-01",
                end="2024-01-03",
                algorithm="dtw_optional",
            ),
        )
