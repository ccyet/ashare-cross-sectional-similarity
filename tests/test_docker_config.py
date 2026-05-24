from __future__ import annotations

from pathlib import Path


def test_dockerfile_runs_streamlit_as_service_with_healthcheck() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "STREAMLIT_SERVER_ADDRESS=0.0.0.0" in dockerfile
    assert "STREAMLIT_SERVER_PORT=8501" in dockerfile
    assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "127.0.0.1:8501/_stcore/health" in dockerfile


def test_docker_compose_defaults_to_host_trend_data_mount() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "${ASHARE_HOST_DATA_DIR:-/Users/a1234/Desktop/trend-backtest/data}:/data" in compose


def test_docker_compose_is_ready_for_long_running_web_service() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "restart: unless-stopped" in compose
    assert "${ASHARE_HOST_TREND_REPO:-/Users/a1234/Desktop/trend-backtest}:/trend-backtest:ro" in compose
    assert "healthcheck:" in compose
    assert "http://127.0.0.1:8501/_stcore/health" in compose
    assert "DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY:-}" in compose
    assert "DASHSCOPE_API_KEY: ${DASHSCOPE_API_KEY:-}" in compose
    assert "OPENAI_API_KEY: ${OPENAI_API_KEY:-}" in compose
    assert "ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}" in compose


def test_env_example_documents_friend_deployment_inputs() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")

    assert "ASHARE_PORT=8502" in env_example
    assert "ASHARE_HOST_DATA_DIR=/Users/a1234/Desktop/trend-backtest/data" in env_example
    assert "ASHARE_HOST_TREND_REPO=/Users/a1234/Desktop/trend-backtest" in env_example
    assert "DEEPSEEK_API_KEY=" in env_example
    assert "DASHSCOPE_API_KEY=" in env_example
    assert "OPENAI_API_KEY=" in env_example
    assert "ANTHROPIC_API_KEY=" in env_example
