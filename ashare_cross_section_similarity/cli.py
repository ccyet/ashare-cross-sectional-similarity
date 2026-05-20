from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

from ashare_cross_section_similarity.benchmark import run_benchmark
from ashare_cross_section_similarity.data import (
    available_symbols,
    import_price_frame,
    load_local_bars,
    read_price_data_file,
)
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.history import HistorySearchConfig, search_history
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    FORWARD_RETURN_WINDOWS,
    search_cross_section,
)
from ashare_cross_section_similarity.similarity_algorithms import ALGORITHM_CHOICES, BASELINE_ALGORITHM
from ashare_cross_section_similarity.universe import (
    fetch_concept_constituents,
    fetch_index_constituents,
    fetch_industry_constituents,
    load_universe_file,
    unique_symbols,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "download":
        return _run_download(args)
    if args.command == "check":
        return _run_check(args)
    if args.command == "import-data":
        return _run_import_data(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "history":
        return _run_history(args)
    return _run_search(args)


def _run_search(args: argparse.Namespace) -> int:
    universe = _resolve_universe(args)
    if not universe:
        raise SystemExit("未得到可搜索范围，请提供 --universe-symbols、--universe-file、--universe-index、--universe-industry、--universe-concept，或准备本地 parquet。")
    symbols_to_load = unique_symbols([args.target_symbol, *universe])
    bars = load_local_bars(
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        symbols=symbols_to_load,
        start=_date_tolerance_load_start(args.start, args.date_tolerance_bars),
        end=_cross_section_load_end(args.end, args.date_tolerance_bars),
    )
    result = search_cross_section(
        bars,
        CrossSectionSearchConfig(
            target_symbol=args.target_symbol,
            universe_symbols=tuple(universe),
            start=args.start,
            end=args.end,
            top_n=args.top_n,
            min_coverage=args.min_coverage,
            path_weight=args.path_weight,
            date_tolerance_bars=args.date_tolerance_bars,
            algorithm=args.algorithm,
        ),
    )
    outcome_columns = [
        f"t_plus_{horizon}_return"
        for horizon in FORWARD_RETURN_WINDOWS
        if f"t_plus_{horizon}_return" in result.results.columns
    ]
    display_columns = [
        "symbol",
        "区间开始",
        "区间结束",
        "日期偏移",
        "覆盖率",
        "综合相似度",
        "路径相似度",
        "特征相似度",
        "区间收益",
        "波动率",
        "最大回撤",
        "K线数量",
        *outcome_columns,
    ]
    print(result.results[display_columns].to_string(index=False) if not result.results.empty else "没有可用结果。")
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.results.to_csv(output_path, index=False)
        print(f"结果已写入：{output_path}")
    return 0


def _run_history(args: argparse.Namespace) -> int:
    forward_windows = _parse_int_list(args.forward_windows)
    bars = load_local_bars(
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        symbols=[args.symbol],
        start=args.start,
        end=args.as_of,
    )
    result = search_history(
        bars,
        HistorySearchConfig(
            symbol=args.symbol,
            as_of=args.as_of,
            window_size=args.window_size,
            forward_windows=tuple(forward_windows),
            candidate_n=args.candidate_n,
            top_n=args.top_n,
            exclusion_bars=args.exclusion_bars,
            nearby_gap_days=args.nearby_gap_days,
            path_weight=args.path_weight,
            algorithm=args.algorithm,
        ),
    )
    outcome_columns = [
        column
        for horizon in forward_windows
        for column in (
            f"t_plus_{horizon}_return",
            f"t_plus_{horizon}_max_drawdown",
            f"t_plus_{horizon}_max_favorable",
        )
        if column in result.results.columns
    ]
    display_columns = [
        "symbol",
        "窗口开始",
        "窗口结束",
        "综合相似度",
        "路径相似度",
        "特征相似度",
        *outcome_columns,
    ]
    print(result.results[display_columns].to_string(index=False) if not result.results.empty else "没有可用结果。")
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.results.to_csv(output_path, index=False)
        print(f"结果已写入：{output_path}")
    return 0


def _run_download(args: argparse.Namespace) -> int:
    symbols = _resolve_download_symbols(args)
    if not symbols:
        raise SystemExit("未得到下载代码，请提供 --symbols 或 universe 参数。")
    result = update_local_bars(
        symbols=symbols,
        timeframe=args.timeframe,
        adjust=args.adjust,
        start=args.start,
        end=args.end,
        trend_repo=args.trend_repo,
        data_root=args.data_root,
        provider=args.provider,
        download_engine=args.download_engine,
    )
    print(result.to_string(index=False))
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"下载日志已写入：{output_path}")
    failed = int((result["status"] == "failed").sum()) if not result.empty else 0
    return 1 if failed else 0


def _run_check(args: argparse.Namespace) -> int:
    symbols = _resolve_download_symbols(args)
    if not symbols:
        raise SystemExit("未得到检查代码，请提供 --symbols 或 universe 参数。")
    result = data_check(
        symbols=symbols,
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        start=args.start,
        end=args.end,
    )
    print(result.to_string(index=False))
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"检查结果已写入：{output_path}")
    return 0


def _run_import_data(args: argparse.Namespace) -> int:
    frame = read_price_data_file(args.input)
    result = import_price_frame(
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        frame=frame,
        fallback_symbol=args.fallback_symbol,
        source_name=Path(args.input).name,
    )
    print(result.to_string(index=False))
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"导入日志已写入：{output_path}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    algorithms = tuple(item.strip() for item in args.algorithms.split(",") if item.strip())
    if not algorithms:
        raise SystemExit("algorithms 至少需要一个算法名。")
    result = run_benchmark(
        cases_path=args.cases,
        algorithms=algorithms,
        output_dir=args.output,
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
    )
    print(result.summary.to_string(index=False))
    print(f"benchmark CSV/HTML 已写入：{Path(args.output).expanduser()}")
    return 1 if (not result.summary.empty and (result.summary["status"] == "failed").any()) else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] not in {"search", "history", "download", "check", "import-data", "benchmark", "-h", "--help"}:
        argv.insert(0, "search")
    parser = argparse.ArgumentParser(description="A股相似阶段搜集：历史时序、横截面、数据抓取、检查")
    subparsers = parser.add_subparsers(dest="command")
    search_parser = subparsers.add_parser("search", help="搜索同一时间窗口内的横截面相似标的")
    _add_common_data_args(search_parser)
    _add_universe_args(search_parser)
    search_parser.add_argument("--target-symbol", required=True, help="目标个股、指数或板块代理代码，如 300750.SZ")
    search_parser.add_argument("--start", required=True, help="目标区间开始时间")
    search_parser.add_argument("--end", required=True, help="目标区间结束时间")
    search_parser.add_argument("--top-n", type=int, default=20)
    search_parser.add_argument("--min-coverage", type=float, default=0.8)
    search_parser.add_argument("--path-weight", type=float, default=0.7)
    search_parser.add_argument("--algorithm", default=BASELINE_ALGORITHM, choices=ALGORITHM_CHOICES, help="相似算法")
    search_parser.add_argument(
        "--date-tolerance-bars",
        type=int,
        default=0,
        help="候选标的窗口允许前后平移的交易日根数；CLI 默认 0，保持严格同日。",
    )
    search_parser.add_argument("--output", default="", help="CSV 输出路径")

    history_parser = subparsers.add_parser("history", help="搜索同一标的的历史相似阶段")
    _add_common_data_args(history_parser)
    history_parser.add_argument("--symbol", required=True, help="目标个股、指数或板块代理代码，如 399006.SZ")
    history_parser.add_argument("--as-of", required=True, help="当前窗口结束日期或时间")
    history_parser.add_argument("--start", default="1900-01-01", help="历史数据读取起点，默认尽量读取全历史")
    history_parser.add_argument("--window-size", type=int, default=20, help="主走势窗口长度")
    history_parser.add_argument("--forward-windows", default="5,20,60", help="后验观察窗口，逗号分隔")
    history_parser.add_argument("--candidate-n", type=int, default=100, help="初筛候选数量")
    history_parser.add_argument("--top-n", type=int, default=10, help="展示数量")
    history_parser.add_argument("--exclusion-bars", type=int, default=20, help="排除当前窗口附近的 K 线数量")
    history_parser.add_argument("--nearby-gap-days", type=int, default=20, help="相邻历史样本最小间隔天数")
    history_parser.add_argument("--path-weight", type=float, default=0.7, help="走势形状在综合相似度中的权重")
    history_parser.add_argument("--algorithm", default=BASELINE_ALGORITHM, choices=ALGORITHM_CHOICES, help="相似算法")
    history_parser.add_argument("--output", default="", help="CSV 输出路径")

    download_parser = subparsers.add_parser("download", help="抓取行情并落地本地 parquet")
    _add_download_data_args(download_parser)
    _add_download_symbol_args(download_parser)

    check_parser = subparsers.add_parser("check", help="检查本地 parquet 覆盖情况")
    _add_common_data_args(check_parser)
    _add_download_symbol_args(check_parser)

    import_parser = subparsers.add_parser("import-data", help="导入符合规范的 csv/parquet 价格数据")
    _add_common_data_args(import_parser)
    import_parser.add_argument("--input", required=True, help="价格数据文件，支持 csv/parquet")
    import_parser.add_argument(
        "--fallback-symbol",
        default="",
        help="当文件没有 symbol/stock_code 列时使用的单一标的代码",
    )
    import_parser.add_argument("--output", default="", help="CSV 导入日志输出路径")

    benchmark_parser = subparsers.add_parser("benchmark", help="运行固定样本算法核验并输出 CSV/HTML 图集")
    _add_common_data_args(benchmark_parser)
    benchmark_parser.add_argument("--cases", required=True, help="benchmark cases yaml/json 文件")
    benchmark_parser.add_argument(
        "--algorithms",
        default="baseline_price_feature,return_shape,hybrid_shape_v2",
        help="逗号分隔算法名",
    )
    benchmark_parser.add_argument("--output", default="outputs/research", help="输出目录")

    parsed = parser.parse_args(argv)
    if parsed.command is None:
        parser.print_help()
        raise SystemExit(2)
    return parsed


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data/market/daily", help="本地行情根目录")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "30m", "15m", "5m", "1m"])
    parser.add_argument("--adjust", default="qfq")


def _add_download_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data/market/daily", help="OpenBB 下载写入的本地行情根目录")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "30m", "15m", "5m", "1m"])
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument(
        "--download-engine",
        default="trend",
        choices=["trend", "openbb", "tdx"],
        help="下载引擎：trend 委托原 trend-backtest；openbb/tdx 直接写入 parquet。",
    )


def _add_universe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--universe-symbols", default="", help="逗号分隔的搜索范围代码")
    parser.add_argument("--universe-file", default="", help="包含证券代码列的 csv/xlsx/parquet 文件")
    parser.add_argument("--universe-index", default="", help="指数代码，如 000300；需要 akshare")
    parser.add_argument("--universe-industry", default="", help="东方财富行业板块名称或代码；需要 akshare")
    parser.add_argument("--universe-concept", default="", help="东方财富概念板块名称或代码；需要 akshare")


def _add_download_symbol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", default="", help="逗号分隔下载/检查代码")
    _add_universe_args(parser)
    parser.add_argument("--start", required=True, help="开始时间")
    parser.add_argument("--end", required=True, help="结束时间")
    parser.add_argument("--output", default="", help="CSV 输出路径")
    parser.add_argument(
        "--trend-repo",
        default=str(default_trend_repo()),
        help="原 trend-backtest 仓库路径，download 会调用其中 scripts/update_data.py",
    )
    parser.add_argument(
        "--provider",
        default="",
        help="下载源；trend 可填 akshare/tdx，openbb 默认 akshare，tdx 可填 PYPlugins/user 路径。",
    )


def _resolve_universe(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.universe_symbols:
        symbols.extend(args.universe_symbols.split(","))
    if args.universe_file:
        symbols.extend(load_universe_file(args.universe_file))
    if args.universe_index:
        symbols.extend(fetch_index_constituents(args.universe_index))
    if args.universe_industry:
        symbols.extend(fetch_industry_constituents(args.universe_industry))
    if args.universe_concept:
        symbols.extend(fetch_concept_constituents(args.universe_concept))
    if not symbols:
        symbols.extend(available_symbols(args.data_root, args.timeframe, args.adjust))
    return unique_symbols(symbols)


def _resolve_download_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(args.symbols.split(","))
    if _has_explicit_universe(args):
        symbols.extend(_resolve_universe(args))
    return unique_symbols(symbols)


def _has_explicit_universe(args: argparse.Namespace) -> bool:
    return any(
        str(getattr(args, name, "")).strip()
        for name in (
            "universe_symbols",
            "universe_file",
            "universe_index",
            "universe_industry",
            "universe_concept",
        )
    )


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise SystemExit("forward-windows 至少需要一个正整数。")
    if any(item <= 0 for item in parsed):
        raise SystemExit("forward-windows 必须为正整数。")
    return parsed


def _forward_stats_load_end(end: str | pd.Timestamp) -> str:
    end_ts = pd.Timestamp(end)
    today = pd.Timestamp.today().normalize()
    if end_ts >= today:
        return end_ts.strftime("%Y-%m-%d")
    return min(end_ts + pd.Timedelta(days=45), today).strftime("%Y-%m-%d")


def _cross_section_load_end(end: str | pd.Timestamp, date_tolerance_bars: int) -> str:
    return _forward_stats_load_end(pd.Timestamp(end) + pd.Timedelta(days=_date_tolerance_calendar_days(date_tolerance_bars)))


def _date_tolerance_load_start(start: str | pd.Timestamp, date_tolerance_bars: int) -> str:
    return (pd.Timestamp(start) - pd.Timedelta(days=_date_tolerance_calendar_days(date_tolerance_bars))).strftime("%Y-%m-%d")


def _date_tolerance_calendar_days(date_tolerance_bars: int) -> int:
    if date_tolerance_bars <= 0:
        return 0
    return max(date_tolerance_bars + 2, int(math.ceil(date_tolerance_bars * 2.2)))
