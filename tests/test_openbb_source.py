from __future__ import annotations

import builtins

import pandas as pd
import pytest

from ashare_cross_section_similarity.openbb_source import fetch_openbb_bars


class _OpenBBOutput:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame


class _HistoricalEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _OpenBBOutput:
        self.calls.append(kwargs)
        return _OpenBBOutput(
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02"],
                    "open": [10, 11],
                    "high": [12, 13],
                    "low": [9, 10],
                    "close": [11, 12],
                    "volume": [1000, 1100],
                }
            )
        )


class _FakeOpenBB:
    def __init__(self) -> None:
        self.historical = _HistoricalEndpoint()
        self.equity = type(
            "Equity",
            (),
            {"price": type("Price", (), {"historical": self.historical})()},
        )()


def test_fetch_openbb_bars_normalizes_output_to_local_schema() -> None:
    fake = _FakeOpenBB()

    out = fetch_openbb_bars(
        symbols=("600519.SH", "000001.SZ"),
        start="2024-01-01",
        end="2024-01-02",
        provider="akshare",
        obb_client=fake,
    )

    assert fake.historical.calls[0]["symbol"] == "600519"
    assert fake.historical.calls[0]["start_date"] == "2024-01-01"
    assert fake.historical.calls[0]["end_date"] == "2024-01-02"
    assert fake.historical.calls[0]["provider"] == "akshare"
    assert out.columns.tolist() == [
        "date",
        "stock_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert out["stock_code"].tolist() == [
        "600519.SH",
        "600519.SH",
        "000001.SZ",
        "000001.SZ",
    ]
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert out["amount"].isna().all()


def test_fetch_openbb_bars_reports_missing_openbb_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openbb":
            raise ImportError("missing openbb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install openbb openbb_akshare"):
        fetch_openbb_bars(
            symbols=("600519.SH",),
            start="2024-01-01",
            end="2024-01-02",
            provider="akshare",
        )
