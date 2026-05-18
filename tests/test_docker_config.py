from __future__ import annotations

from pathlib import Path


def test_docker_compose_defaults_to_host_trend_data_mount() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "${ASHARE_HOST_DATA_DIR:-/Users/a1234/Desktop/trend-backtest/data}:/data" in compose
