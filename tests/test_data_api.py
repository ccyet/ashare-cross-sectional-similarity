from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from ashare_cross_section_similarity.data_api import DataApiSettings, create_app


def test_data_api_rejects_requests_outside_small_scope(tmp_path: Path) -> None:
    app = create_app(DataApiSettings(data_root=tmp_path, max_symbols_per_request=1, max_days_per_request=7))
    client = TestClient(app)

    response = client.get(
        "/api/v1/bars",
        params={
            "symbols": "000001.SZ,600519.SH",
            "start": "2024-01-01",
            "end": "2024-01-05",
            "source": "local",
        },
    )

    assert response.status_code == 400
    assert "单次最多请求 1 个标的" in response.json()["detail"]


def test_data_api_fetches_akshare_before_returning_bars(tmp_path: Path) -> None:
    app = create_app(DataApiSettings(data_root=tmp_path, timeframe="1d", adjust="qfq"))
    client = TestClient(app)
    bars = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "stock_code": ["000001.SZ"],
            "open": [1.0],
            "high": [1.2],
            "low": [0.9],
            "close": [1.1],
            "volume": [100.0],
            "amount": [1000.0],
        }
    )
    download = pd.DataFrame(
        [{"symbol": "000001.SZ", "status": "success", "rows": 1, "new_rows": 1, "message": ""}]
    )

    with (
        patch("ashare_cross_section_similarity.data_api.update_local_bars", return_value=download) as update,
        patch("ashare_cross_section_similarity.data_api.load_local_bars", return_value=bars) as load,
    ):
        response = client.get(
            "/api/v1/bars",
            params={
                "symbols": "000001",
                "start": "2024-01-01",
                "end": "2024-01-02",
                "source": "akshare",
            },
        )

    assert response.status_code == 200
    update.assert_called_once_with(
        symbols=("000001.SZ",),
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-02",
        data_root=tmp_path,
        download_engine="akshare",
        provider="",
    )
    load.assert_called_once()
    payload = response.json()
    assert payload["meta"]["source"] == "akshare"
    assert payload["meta"]["symbols"] == ["000001.SZ"]
    assert payload["records"][0]["stock_code"] == "000001.SZ"
    assert payload["download_status"][0]["status"] == "success"


def test_data_api_fetches_tdx_with_configured_local_path(tmp_path: Path) -> None:
    app = create_app(DataApiSettings(data_root=tmp_path, tdx_path="/Applications/Tdx/PYPlugins/user"))
    client = TestClient(app)

    with (
        patch("ashare_cross_section_similarity.data_api.update_local_bars", return_value=pd.DataFrame()) as update,
        patch(
            "ashare_cross_section_similarity.data_api.load_local_bars",
            return_value=pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]),
        ),
    ):
        response = client.get(
            "/api/v1/bars",
            params={
                "symbols": "600519.SH",
                "start": "2024-01-01",
                "end": "2024-01-02",
                "source": "tdx",
            },
        )

    assert response.status_code == 200
    update.assert_called_once_with(
        symbols=("600519.SH",),
        timeframe="1d",
        adjust="qfq",
        start="2024-01-01",
        end="2024-01-02",
        data_root=tmp_path,
        download_engine="tdx",
        provider="/Applications/Tdx/PYPlugins/user",
    )


def test_data_api_reports_missing_tdx_configuration(tmp_path: Path) -> None:
    app = create_app(DataApiSettings(data_root=tmp_path, tdx_path=""))
    client = TestClient(app)

    response = client.get(
        "/api/v1/bars",
        params={
            "symbols": "600519.SH",
            "start": "2024-01-01",
            "end": "2024-01-02",
            "source": "tdx",
        },
    )

    assert response.status_code == 400
    assert "TDX 数据源需要配置" in response.json()["detail"]
