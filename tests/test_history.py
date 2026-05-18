from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ashare_cross_section_similarity.data import inclusive_end_timestamp
from ashare_cross_section_similarity.features import (
    FEATURE_COLUMNS,
    max_drawdown,
    normalized_close_path,
    window_features,
    z_normalize,
)
from ashare_cross_section_similarity.history import (
    HistorySearchConfig,
    _filter_nearby_history_windows,
    search_history,
)
from ashare_cross_section_similarity.similarity import _prepare_bars, _score_results
from ashare_cross_section_similarity.universe import normalize_symbol


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


def test_search_history_finds_same_symbol_prior_similar_windows() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10, 11, 12, 11, 13,
            20, 19, 18, 17, 16,
            30, 33, 36, 33, 39,
            40, 41, 42, 43, 44,
        ],
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(2,),
            top_n=2,
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    assert result.symbol == "000001.SZ"
    assert result.current_window["date"].min() == pd.Timestamp("2024-01-11")
    assert result.results["窗口开始"].tolist()[0] == pd.Timestamp("2024-01-01")
    assert result.historical_windows[0]["date"].min() == pd.Timestamp("2024-01-01")
    assert result.results["综合相似度"].iloc[0] > result.results["综合相似度"].iloc[1]
    assert "t_plus_2_return" in result.results.columns


def test_search_history_does_not_use_data_after_as_of_for_matching() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10, 11, 12, 11, 13,
            20, 19, 18, 17, 16,
            30, 33, 36, 33, 39,
            30, 33, 36, 33, 39,
        ],
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-15",
            window_size=5,
            forward_windows=(2,),
            top_n=10,
            exclusion_bars=0,
        ),
    )

    assert result.results["窗口开始"].max() < pd.Timestamp("2024-01-11")


def test_search_history_uses_explicit_window_start_instead_of_backsolving() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10, 11, 12, 11, 13,
            15, 14,
            20, 22, 24, 22, 26,
            30, 31, 32,
        ],
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-12",
            window_size=2,
            window_start="2024-01-08",
            forward_windows=(1,),
            top_n=1,
            exclusion_bars=0,
            nearby_gap_days=0,
        ),
    )

    assert result.window_size == 5
    assert result.current_window["date"].min() == pd.Timestamp("2024-01-08")
    assert result.current_window["date"].max() == pd.Timestamp("2024-01-12")
    assert result.results["K线数量"].iloc[0] == 5
    assert result.results["窗口开始"].iloc[0] == pd.Timestamp("2024-01-01")


def test_search_history_rejects_too_short_explicit_window() -> None:
    bars = _bars("000001.SZ", [10, 11, 12])

    with pytest.raises(ValueError, match="选定区间"):
        search_history(
            bars,
            HistorySearchConfig(
                symbol="000001.SZ",
                as_of="2024-01-02",
                window_size=2,
                window_start="2024-01-02",
            ),
        )


def test_search_history_date_only_as_of_includes_full_intraday_session() -> None:
    bars = pd.DataFrame(
        {
            "date": [
                "2024-01-01 10:00:00",
                "2024-01-01 14:30:00",
                "2024-01-02 10:00:00",
            ],
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [1, 1, 1],
            "amount": [1, 1, 1],
        }
    )

    result = search_history(
        bars,
        HistorySearchConfig(
            symbol="000001.SZ",
            as_of="2024-01-01",
            window_size=2,
            forward_windows=(),
            exclusion_bars=0,
        ),
    )

    assert result.current_window["date"].tolist() == [
        pd.Timestamp("2024-01-01 10:00:00"),
        pd.Timestamp("2024-01-01 14:30:00"),
    ]


def test_search_history_rejects_invalid_ranking_parameters() -> None:
    bars = _bars("000001.SZ", [10, 11, 12])

    with pytest.raises(ValueError, match="top_n"):
        search_history(
            bars,
            HistorySearchConfig(symbol="000001.SZ", as_of="2024-01-03", window_size=2, top_n=0),
        )

    with pytest.raises(ValueError, match="forward_windows"):
        search_history(
            bars,
            HistorySearchConfig(
                symbol="000001.SZ",
                as_of="2024-01-03",
                window_size=2,
                forward_windows=(0,),
            ),
        )


def test_search_history_matches_legacy_window_loop() -> None:
    bars = _bars(
        "000001.SZ",
        [
            10, 11, 13, 12, 14,
            16, 15, 17, 20, 18,
            21, 23, 22, 24, 27,
            30, 28, 31, 33, 32,
            35, 38, 37, 40, 42,
            44, 43, 46, 48, 47,
        ],
    )
    config = HistorySearchConfig(
        symbol="000001.SZ",
        as_of="2024-01-28",
        window_size=5,
        forward_windows=(2, 4),
        candidate_n=8,
        top_n=5,
        exclusion_bars=2,
        nearby_gap_days=0,
        path_weight=0.65,
    )

    actual = search_history(bars, config).results
    expected = _legacy_history_results(bars, config)

    assert actual["窗口开始"].tolist() == expected["窗口开始"].tolist()
    assert actual["窗口结束"].tolist() == expected["窗口结束"].tolist()
    numeric_columns = [column for column in actual.columns if column not in {"symbol", "窗口开始", "窗口结束"}]
    np.testing.assert_allclose(
        actual[numeric_columns].to_numpy(dtype=float),
        expected[numeric_columns].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_nearby_history_windows_keep_most_similar_sample() -> None:
    frame = pd.DataFrame(
        [
            {
                "窗口开始": pd.Timestamp("2016-12-17"),
                "窗口结束": pd.Timestamp("2017-01-08"),
                "综合相似度": 0.95,
            },
            {
                "窗口开始": pd.Timestamp("2016-12-15"),
                "窗口结束": pd.Timestamp("2017-01-06"),
                "综合相似度": 0.90,
            },
            {
                "窗口开始": pd.Timestamp("2017-03-01"),
                "窗口结束": pd.Timestamp("2017-03-20"),
                "综合相似度": 0.80,
            },
        ]
    )

    filtered = _filter_nearby_history_windows(frame, top_n=3, min_gap_days=20)

    assert filtered["窗口开始"].tolist() == [
        pd.Timestamp("2016-12-17"),
        pd.Timestamp("2017-03-01"),
    ]


def _legacy_history_results(bars: pd.DataFrame, config: HistorySearchConfig) -> pd.DataFrame:
    prepared = _prepare_bars(bars)
    symbol = normalize_symbol(config.symbol)
    prepared = prepared.loc[prepared["stock_code"] == symbol].sort_values("date").reset_index(drop=True)
    as_of = inclusive_end_timestamp(config.as_of)
    available = prepared.loc[prepared["date"] <= as_of].copy()
    as_of_index = int(available.index[-1])
    current_start = as_of_index - config.window_size + 1
    current_window = prepared.iloc[current_start : as_of_index + 1].reset_index(drop=True)
    target_path = z_normalize(normalized_close_path(current_window))
    target_features = window_features(current_window)
    max_forward = max(config.forward_windows) if config.forward_windows else 0
    rows: list[dict[str, object]] = []

    latest_end = current_start - max(config.exclusion_bars, 0) - 1
    latest_start = latest_end - config.window_size + 1
    for start in range(0, max(0, latest_start + 1)):
        end = start + config.window_size - 1
        if end + max_forward > as_of_index:
            continue
        candidate = prepared.iloc[start : end + 1].reset_index(drop=True)
        candidate_path = z_normalize(normalized_close_path(candidate))
        features = window_features(candidate)
        row: dict[str, object] = {
            "_candidate_index": len(rows),
            "symbol": symbol,
            "窗口开始": candidate["date"].min(),
            "窗口结束": candidate["date"].max(),
            "K线数量": int(len(candidate)),
            "路径距离": float(np.linalg.norm(target_path - candidate_path) / math.sqrt(config.window_size)),
        }
        for column in FEATURE_COLUMNS:
            row[column] = features[column]
            row[f"feature_diff::{column}"] = abs(features[column] - target_features[column])
        row.update(_legacy_forward_outcomes(prepared, end, config.forward_windows))
        rows.append(row)

    scored = _score_results(pd.DataFrame(rows), config.path_weight)
    scored = scored.sort_values(["综合相似度", "路径相似度"], ascending=False).head(config.candidate_n)
    return (
        _filter_nearby_history_windows(scored, top_n=config.top_n, min_gap_days=config.nearby_gap_days)
        .drop(columns=["_candidate_index"])
        .reset_index(drop=True)
    )


def _legacy_forward_outcomes(
    bars: pd.DataFrame,
    window_end_index: int,
    forward_windows: tuple[int, ...],
) -> dict[str, float]:
    outcomes: dict[str, float] = {}
    base_close = float(bars.iloc[window_end_index]["close"])
    for horizon in forward_windows:
        target_index = window_end_index + horizon
        if target_index >= len(bars) or base_close == 0:
            outcomes[f"t_plus_{horizon}_return"] = float("nan")
            outcomes[f"t_plus_{horizon}_max_drawdown"] = float("nan")
            outcomes[f"t_plus_{horizon}_max_favorable"] = float("nan")
            continue
        future = bars.iloc[window_end_index + 1 : target_index + 1]
        closes = pd.to_numeric(future["close"], errors="coerce").astype(float).to_numpy()
        outcomes[f"t_plus_{horizon}_return"] = float(closes[-1] / base_close - 1.0)
        outcomes[f"t_plus_{horizon}_max_drawdown"] = max_drawdown(np.r_[base_close, closes])
        outcomes[f"t_plus_{horizon}_max_favorable"] = float(np.nanmax(closes / base_close - 1.0))
    return outcomes
