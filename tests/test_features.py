from __future__ import annotations

import warnings

import pandas as pd

from ashare_cross_section_similarity.features import window_features


def test_window_features_constant_series_corr_is_warning_free() -> None:
    bars = pd.DataFrame(
        {
            "close": [1.0, 1.0, 1.0],
            "volume": [100.0, 100.0, 100.0],
            "amount": [1000.0, 1000.0, 1000.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        features = window_features(bars)

    assert features["量价相关"] == 0.0
