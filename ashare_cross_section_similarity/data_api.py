from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.data import load_local_bars
from ashare_cross_section_similarity.downloader import data_check, update_local_bars
from ashare_cross_section_similarity.tdx_source import TDX_TQCENTER_ENV_VAR
from ashare_cross_section_similarity.universe import unique_symbols

try:
    from fastapi import FastAPI, HTTPException, Query
except ImportError as exc:  # pragma: no cover - exercised only when optional deps are absent
    raise RuntimeError("数据 API 需要安装 server 依赖：pip install fastapi uvicorn") from exc

DATA_API_SOURCES = {"local", "akshare", "tdx"}


@dataclass(frozen=True)
class DataApiSettings:
    data_root: str | Path = "data/market/daily"
    timeframe: str = "1d"
    adjust: str = "qfq"
    tdx_path: str = ""
    max_symbols_per_request: int = 30
    max_days_per_request: int = 370


def create_app(settings: DataApiSettings | None = None) -> FastAPI:
    config = settings or DataApiSettings(tdx_path=os.getenv(TDX_TQCENTER_ENV_VAR, ""))
    app = FastAPI(
        title="A股相似阶段数据 API",
        version="0.1.0",
        description="面向 Qt 和局域网小范围调用的 A 股 K 线数据 API。",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        tdx_path = _configured_tdx_path(config)
        return {
            "status": "ok",
            "sources": {
                "local": {"mode": "read_parquet"},
                "akshare": {"mode": "auto_fetch_then_read_parquet"},
                "tdx": {
                    "mode": "local_tqcenter_then_read_parquet",
                    "configured": bool(tdx_path),
                    "path": tdx_path,
                },
            },
            "limits": {
                "max_symbols_per_request": config.max_symbols_per_request,
                "max_days_per_request": config.max_days_per_request,
            },
        }

    @app.get("/api/v1/coverage")
    def coverage(
        symbols: str = Query(..., description="逗号分隔代码，如 000001.SZ,600519.SH"),
        start: str = Query(...),
        end: str = Query(...),
        timeframe: str | None = Query(None),
        adjust: str | None = Query(None),
    ) -> dict[str, object]:
        parsed_symbols, start_ts, end_ts = _validate_small_request(symbols, start, end, config)
        frame = data_check(
            symbols=parsed_symbols,
            data_root=config.data_root,
            timeframe=timeframe or config.timeframe,
            adjust=adjust or config.adjust,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
        )
        return {
            "meta": _response_meta("local", parsed_symbols, start_ts, end_ts, timeframe or config.timeframe, adjust or config.adjust),
            "records": _records(frame),
        }

    @app.get("/api/v1/bars")
    def bars(
        symbols: str = Query(..., description="逗号分隔代码，如 000001.SZ,600519.SH"),
        start: str = Query(...),
        end: str = Query(...),
        source: str = Query("local", description="local, akshare 或 tdx"),
        timeframe: str | None = Query(None),
        adjust: str | None = Query(None),
        tdx_path: str | None = Query(None, description="TDX 的 PYPlugins/user 目录；为空时读取服务配置或环境变量"),
    ) -> dict[str, object]:
        parsed_symbols, start_ts, end_ts = _validate_small_request(symbols, start, end, config)
        resolved_source = _normalize_source(source)
        resolved_timeframe = timeframe or config.timeframe
        resolved_adjust = adjust or config.adjust
        download_status = pd.DataFrame()
        if resolved_source != "local":
            download_status = _fetch_to_local(
                source=resolved_source,
                symbols=parsed_symbols,
                start=start_ts.strftime("%Y-%m-%d"),
                end=end_ts.strftime("%Y-%m-%d"),
                timeframe=resolved_timeframe,
                adjust=resolved_adjust,
                settings=config,
                request_tdx_path=tdx_path or "",
            )
        frame = load_local_bars(
            data_root=config.data_root,
            timeframe=resolved_timeframe,
            adjust=resolved_adjust,
            symbols=parsed_symbols,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
        )
        return {
            "meta": _response_meta(resolved_source, parsed_symbols, start_ts, end_ts, resolved_timeframe, resolved_adjust),
            "download_status": _records(download_status),
            "records": _records(frame),
        }

    return app


def _fetch_to_local(
    *,
    source: str,
    symbols: tuple[str, ...],
    start: str,
    end: str,
    timeframe: str,
    adjust: str,
    settings: DataApiSettings,
    request_tdx_path: str,
) -> pd.DataFrame:
    if source == "akshare":
        provider = ""
        download_engine = "akshare"
    elif source == "tdx":
        provider = request_tdx_path or _configured_tdx_path(settings)
        if not provider:
            raise HTTPException(
                status_code=400,
                detail=f"TDX 数据源需要配置 {TDX_TQCENTER_ENV_VAR} 或请求参数 tdx_path。",
            )
        download_engine = "tdx"
    else:  # pragma: no cover - guarded by _normalize_source
        raise HTTPException(status_code=400, detail=f"不支持的数据源：{source}")

    try:
        frame = update_local_bars(
            symbols=symbols,
            timeframe=timeframe,
            adjust=adjust,
            start=start,
            end=end,
            data_root=settings.data_root,
            download_engine=download_engine,
            provider=provider,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if _all_downloads_failed(frame):
        message = "；".join(str(item) for item in frame.get("message", pd.Series(dtype=object)).dropna().tolist())
        raise HTTPException(status_code=502, detail=message or f"{source} 数据源抓取失败。")
    return frame


def _validate_small_request(
    symbols: str,
    start: str,
    end: str,
    settings: DataApiSettings,
) -> tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    parsed_symbols = tuple(unique_symbols(symbols.replace("\n", ",").split(",")))
    if not parsed_symbols:
        raise HTTPException(status_code=400, detail="至少需要一个标的代码。")
    if len(parsed_symbols) > settings.max_symbols_per_request:
        raise HTTPException(status_code=400, detail=f"单次最多请求 {settings.max_symbols_per_request} 个标的。")
    try:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="start/end 必须是可解析日期。") from exc
    if end_ts < start_ts:
        raise HTTPException(status_code=400, detail="end 不能早于 start。")
    requested_days = int((end_ts - start_ts).days) + 1
    if requested_days > settings.max_days_per_request:
        raise HTTPException(status_code=400, detail=f"单次最多请求 {settings.max_days_per_request} 个自然日。")
    return parsed_symbols, start_ts, end_ts


def _normalize_source(source: str) -> str:
    normalized = source.strip().lower()
    if normalized not in DATA_API_SOURCES:
        raise HTTPException(status_code=400, detail="source 仅支持 local、akshare 或 tdx。")
    return normalized


def _configured_tdx_path(settings: DataApiSettings) -> str:
    return (settings.tdx_path or os.getenv(TDX_TQCENTER_ENV_VAR, "")).strip()


def _all_downloads_failed(frame: pd.DataFrame) -> bool:
    if frame.empty or "status" not in frame.columns:
        return False
    statuses = frame["status"].astype(str).str.lower()
    return bool(len(statuses) and (statuses == "failed").all())


def _response_meta(
    source: str,
    symbols: tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe: str,
    adjust: str,
) -> dict[str, object]:
    return {
        "source": source,
        "symbols": list(symbols),
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "timeframe": timeframe,
        "adjust": adjust,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    normalized = normalized.replace([math.inf, -math.inf], pd.NA)
    normalized = normalized.astype(object).where(pd.notna(normalized), None)
    return normalized.to_dict(orient="records")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 A 股小范围数据 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", default="data/market/daily")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "30m", "15m", "5m", "1m"])
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--tdx-path", default=os.getenv(TDX_TQCENTER_ENV_VAR, ""))
    parser.add_argument("--max-symbols", type=int, default=30)
    parser.add_argument("--max-days", type=int, default=370)
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("数据 API 需要安装 uvicorn：pip install uvicorn") from exc
    app = create_app(
        DataApiSettings(
            data_root=args.data_root,
            timeframe=args.timeframe,
            adjust=args.adjust,
            tdx_path=args.tdx_path,
            max_symbols_per_request=args.max_symbols,
            max_days_per_request=args.max_days,
        )
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
