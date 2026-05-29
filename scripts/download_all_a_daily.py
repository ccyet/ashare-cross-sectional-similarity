from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ashare_cross_section_similarity.downloader import (  # noqa: E402
    data_check,
    default_trend_repo,
    plan_incremental_downloads,
    update_local_bars,
)
from ashare_cross_section_similarity.tdx_source import fetch_tdx_stock_symbols  # noqa: E402
from ashare_cross_section_similarity.universe import (  # noqa: E402
    fetch_all_a_symbols,
    symbols_with_analysis_indexes,
    unique_symbols,
)

DEFAULT_START = "1990-01-01"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stock_symbols = _fetch_stock_symbols_for_engine(args.download_engine, args.provider)
    all_symbols = _download_universe(
        stock_symbols,
        include_indexes=bool(args.include_indexes),
        extra_symbols=args.extra_symbols,
    )
    if args.symbols_output:
        _write_symbols(args.symbols_output, all_symbols)
    symbols = all_symbols
    if args.limit:
        symbols = symbols[: args.limit]
    if args.dry_run:
        print(f"下载范围数：{len(all_symbols):,}")
        if args.limit:
            print(f"本次限制处理：{len(symbols):,}")
        print(f"前 10 个：{', '.join(all_symbols[:10])}")
        return 0
    result = download_all_a_daily(
        symbols=symbols,
        trend_repo=Path(args.trend_repo),
        data_root=Path(args.data_root),
        adjust=args.adjust,
        start=args.start,
        end=args.end,
        provider=args.provider,
        download_engine=args.download_engine,
        batch_size=args.batch_size,
        output=Path(args.output),
        skip_available=args.skip_available,
        incremental_latest_only=bool(args.incremental_latest_only),
        sleep_seconds=args.sleep,
    )
    failed = int((result["status"] == "failed").sum()) if not result.empty else 0
    missing = int(result["status"].isin(["missing_file", "missing_window", "read_error"]).sum()) if not result.empty else 0
    print(f"完成：{len(result):,} 个；failed={failed:,}；missing/read_error={missing:,}；日志={args.output}")
    return 1 if failed else 0


def download_all_a_daily(
    *,
    symbols: Iterable[str],
    trend_repo: Path,
    data_root: Path,
    adjust: str,
    start: str,
    end: str,
    provider: str,
    download_engine: str,
    batch_size: int,
    output: Path,
    skip_available: bool = False,
    incremental_latest_only: bool = False,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    normalized = unique_symbols(symbols)
    start_by_symbol: dict[str, str] = {}
    if incremental_latest_only:
        plan = plan_incremental_downloads(
            symbols=normalized,
            data_root=data_root,
            timeframe="1d",
            adjust=adjust,
            start=start,
            end=end,
        )
        if not plan.empty:
            required = plan.loc[plan["download_required"].fillna(False)]
            start_by_symbol = {
                str(row["symbol"]): str(row["download_start"])
                for _, row in required.iterrows()
            }
        original_count = len(normalized)
        normalized = [symbol for symbol in normalized if symbol in start_by_symbol]
        print(f"增量补最新：跳过 {original_count - len(normalized):,}；待下载：{len(normalized):,}")
    elif skip_available:
        before = data_check(
            symbols=normalized,
            data_root=data_root,
            timeframe="1d",
            adjust=adjust,
            start=start,
            end=end,
        )
        available = set(before.loc[before["status"] == "available", "symbol"].tolist())
        normalized = [symbol for symbol in normalized if symbol not in available]
        print(f"跳过已覆盖：{len(available):,}；待下载：{len(normalized):,}")
    if not normalized:
        return pd.DataFrame(columns=["symbol", "status", "rows", "start", "end", "message"])
    download_groups = _group_symbols_by_download_start(normalized, start_by_symbol, default_start=start)
    batches = [
        (download_start, batch)
        for download_start, group_symbols in download_groups
        for batch in _batched(group_symbols, batch_size)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    frames: list[pd.DataFrame] = []
    for index, (download_start, batch) in enumerate(batches, start=1):
        print(f"[{index}/{len(batches)}] 下载 {len(batch)} 个：{batch[0]} ... {batch[-1]}，起点 {download_start}")
        download_result = update_local_bars(
            symbols=batch,
            timeframe="1d",
            adjust=adjust,
            start=download_start,
            end=end,
            trend_repo=trend_repo,
            data_root=data_root,
            provider=provider,
            download_engine=download_engine,
        )
        checked = data_check(
            symbols=batch,
            data_root=data_root,
            timeframe="1d",
            adjust=adjust,
            start=download_start if incremental_latest_only else start,
            end=end,
        )
        merged = merge_download_check(download_result, checked)
        _append_csv(output, merged)
        frames.append(merged)
        if sleep_seconds > 0 and index < len(batches):
            time.sleep(sleep_seconds)
    return pd.concat(frames, ignore_index=True)


def _group_symbols_by_download_start(
    symbols: list[str],
    start_by_symbol: dict[str, str],
    *,
    default_start: str,
) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for symbol in symbols:
        download_start = start_by_symbol.get(symbol, default_start)
        groups.setdefault(download_start, []).append(symbol)
    return list(groups.items())


def _download_universe(symbols: Iterable[object], *, include_indexes: bool, extra_symbols: str) -> list[str]:
    return symbols_with_analysis_indexes(
        symbols,
        include_indexes=include_indexes,
        extra_symbols=_split_symbol_text(extra_symbols),
    )


def _fetch_stock_symbols_for_engine(download_engine: str, provider: str) -> list[str]:
    if download_engine == "tdx":
        return fetch_tdx_stock_symbols(tqcenter_path=provider)
    return fetch_all_a_symbols()


def _split_symbol_text(value: str) -> list[str]:
    text = str(value or "")
    for separator in ("，", "、", ";", "；", "\n", "\t", " "):
        text = text.replace(separator, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def merge_download_check(download_result: pd.DataFrame, checked: pd.DataFrame) -> pd.DataFrame:
    if checked.empty:
        return download_result.copy()
    result = checked.copy()
    download_rows = download_result.set_index("symbol") if not download_result.empty and "symbol" in download_result.columns else pd.DataFrame()
    for index, row in result.iterrows():
        symbol = row["symbol"]
        if symbol not in download_rows.index:
            continue
        download_row = download_rows.loc[symbol]
        if isinstance(download_row, pd.DataFrame):
            download_row = download_row.iloc[0]
        download_status = str(download_row.get("status", ""))
        download_message = str(download_row.get("message", ""))
        if download_status == "failed":
            result.loc[index, "status"] = "failed"
            result.loc[index, "message"] = download_message
        elif row["status"] != "available":
            check_message = str(row.get("message", ""))
            result.loc[index, "message"] = f"下载后仍未覆盖：{check_message}".rstrip("：")
    return result


def _batched(symbols: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size 至少需要 1。")
    for start in range(0, len(symbols), batch_size):
        yield symbols[start : start + batch_size]


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def _write_symbols(path: str, symbols: list[str]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(output, index=False, encoding="utf-8-sig")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载全 A 股票历史以来全部日线数据")
    parser.add_argument("--trend-repo", default=str(default_trend_repo()), help="trend-backtest 仓库路径")
    parser.add_argument("--data-root", default=str(default_trend_repo() / "data" / "market" / "daily"), help="行情根目录")
    parser.add_argument("--adjust", default="qfq", help="复权目录，默认 qfq")
    parser.add_argument("--start", default=DEFAULT_START, help="历史起点，默认 1990-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="结束日期，默认今天")
    parser.add_argument("--download-engine", choices=["trend", "openbb", "tdx"], default="trend", help="下载引擎")
    parser.add_argument("--provider", default="", help="数据源；trend 可填 akshare/tdx，openbb 默认 akshare，tdx 可填 PYPlugins/user 路径")
    parser.add_argument("--batch-size", type=int, default=100, help="每批下载股票数")
    parser.add_argument("--output", default="outputs/all_a_daily_download_log.csv", help="下载和覆盖检查日志")
    parser.add_argument("--symbols-output", default="", help="可选：导出全 A 股票列表")
    parser.add_argument("--no-indexes", dest="include_indexes", action="store_false", help="不额外加入常用指数代理")
    parser.add_argument("--extra-symbols", default="", help="额外下载代码，逗号或换行分隔，如 399006.SZ,000300.SH")
    parser.add_argument("--skip-available", action="store_true", help="下载前跳过已覆盖区间的股票")
    parser.add_argument(
        "--incremental-latest-only",
        action="store_true",
        help="只补齐本地最后一根 K 线之后到结束日期的数据；股票清单仍以下载引擎提供为准。",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="批次间暂停秒数")
    parser.add_argument("--limit", type=int, default=0, help="调试用，仅下载前 N 个")
    parser.add_argument("--dry-run", action="store_true", help="只获取并打印下载范围，不下载")
    parser.set_defaults(include_indexes=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
