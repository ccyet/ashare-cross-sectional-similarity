from __future__ import annotations

import argparse
from pathlib import Path

from ashare_cross_section_similarity.data import available_symbols, load_local_bars
from ashare_cross_section_similarity.similarity import CrossSectionSearchConfig, search_cross_section
from ashare_cross_section_similarity.universe import (
    fetch_concept_constituents,
    fetch_index_constituents,
    fetch_industry_constituents,
    load_universe_file,
    unique_symbols,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    universe = _resolve_universe(args)
    if not universe:
        raise SystemExit("未得到可搜索范围，请提供 --universe-symbols、--universe-file、--universe-index、--universe-industry、--universe-concept，或准备本地 parquet。")
    symbols_to_load = unique_symbols([args.target_symbol, *universe])
    bars = load_local_bars(
        data_root=args.data_root,
        timeframe=args.timeframe,
        adjust=args.adjust,
        symbols=symbols_to_load,
        start=args.start,
        end=args.end,
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
        ),
    )
    display_columns = [
        "symbol",
        "综合相似度",
        "路径相似度",
        "特征相似度",
        "区间收益",
        "波动率",
        "最大回撤",
        "K线数量",
    ]
    print(result.results[display_columns].to_string(index=False) if not result.results.empty else "没有可用结果。")
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.results.to_csv(output_path, index=False)
        print(f"结果已写入：{output_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A股同一时间窗口横截面相似阶段搜索")
    parser.add_argument("--data-root", default="data/market/daily", help="本地行情根目录，通常指向 trend-backtest/data/market/daily")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "30m", "15m", "5m", "1m"])
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--target-symbol", required=True, help="目标个股、指数或板块代理代码，如 300750.SZ")
    parser.add_argument("--start", required=True, help="目标区间开始时间")
    parser.add_argument("--end", required=True, help="目标区间结束时间")
    parser.add_argument("--universe-symbols", default="", help="逗号分隔的搜索范围代码")
    parser.add_argument("--universe-file", default="", help="包含证券代码列的 csv/xlsx/parquet 文件")
    parser.add_argument("--universe-index", default="", help="指数代码，如 000300；需要 akshare")
    parser.add_argument("--universe-industry", default="", help="东方财富行业板块名称或代码；需要 akshare")
    parser.add_argument("--universe-concept", default="", help="东方财富概念板块名称或代码；需要 akshare")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--path-weight", type=float, default=0.7)
    parser.add_argument("--output", default="", help="CSV 输出路径")
    return parser.parse_args(argv)


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
