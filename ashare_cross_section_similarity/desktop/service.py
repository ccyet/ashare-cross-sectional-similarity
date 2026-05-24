from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from ashare_cross_section_similarity.data import available_symbols, import_price_frame, load_local_bars, read_price_data_file
from ashare_cross_section_similarity.data_manager import migrate_kline_data, plan_kline_migration
from ashare_cross_section_similarity.downloader import data_check, default_trend_repo, update_local_bars
from ashare_cross_section_similarity.history import HistorySearchConfig, HistorySearchResult, search_history
from ashare_cross_section_similarity.review import (
    ReviewConfig,
    ReviewResult,
    analyze_price_review,
    build_comparison_stats,
    build_equal_weight_series,
    build_video_script_profile,
    rank_review_results,
)
from ashare_cross_section_similarity.similarity import (
    CrossSectionSearchConfig,
    CrossSectionSearchResult,
    search_cross_section,
)
from ashare_cross_section_similarity.similarity_algorithms import BASELINE_ALGORITHM
from ashare_cross_section_similarity.tdx_source import fetch_tdx_stock_symbols
from ashare_cross_section_similarity.universe import (
    DEFAULT_ANALYSIS_INDEX_SYMBOLS,
    fetch_all_a_symbols,
    load_universe_file,
    normalize_symbol,
    symbols_with_analysis_indexes,
    unique_symbols,
)
from ashare_cross_section_similarity import universe as universe_module

DATA_MODES = {"local", "api"}
DATA_SOURCES = {"local", "akshare", "tdx"}
DEFAULT_LOAD_START = "1900-01-01"
SCRIPT_BENCHMARK_SYMBOL = "000300.SH"
DOWNLOAD_REQUIRED_STATUSES = {"missing_file", "missing_window", "partial_window", "read_error"}
FALLBACK_REVIEW_ETFS = (
    ("510300.SH", "沪深300ETF", 10_000_000.0, "沪深300"),
    ("510500.SH", "中证500ETF", 9_500_000.0, "中证500"),
    ("512100.SH", "中证1000ETF", 9_000_000.0, "中证1000"),
    ("159915.SZ", "创业板ETF", 8_500_000.0, "创业板"),
    ("588000.SH", "科创50ETF", 8_000_000.0, "科创50"),
    ("159995.SZ", "芯片ETF", 7_500_000.0, "半导体"),
    ("512480.SH", "半导体ETF", 7_000_000.0, "半导体"),
    ("512010.SH", "医药ETF", 6_500_000.0, "医药"),
    ("515030.SH", "新能源车ETF", 6_000_000.0, "新能源车"),
    ("512660.SH", "军工ETF", 5_500_000.0, "军工"),
    ("159928.SZ", "消费ETF", 5_000_000.0, "消费"),
    ("512690.SH", "酒ETF", 4_500_000.0, "消费"),
)


def default_desktop_data_root() -> Path:
    trend_root = default_trend_repo() / "data" / "market" / "daily"
    return trend_root if trend_root.exists() else Path("data/market/daily")


def parse_symbol_list(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_symbols = value.replace("\n", ",").replace(" ", ",").split(",")
    else:
        raw_symbols = list(value)
    return tuple(unique_symbols(raw_symbols))


def import_symbol_file(path: str | Path) -> tuple[str, ...]:
    return tuple(load_universe_file(path))


def export_frame_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path).expanduser()
    if output.suffix.lower() != ".csv":
        output = output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _valid_bar_dates(bars: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(bars.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna().sort_values()


@dataclass(frozen=True)
class DesktopAppConfig:
    data_root: str | Path = default_desktop_data_root()
    timeframe: str = "1d"
    adjust: str = "qfq"
    data_mode: str = "local"
    data_source: str = "local"
    api_base_url: str = "http://127.0.0.1:8765"
    tdx_path: str = ""


@dataclass(frozen=True)
class HistoryRequest:
    symbol: str
    as_of: str
    window_size: int = 20
    window_start: str | None = None
    forward_windows: tuple[int, ...] = (5, 20, 60)
    candidate_n: int = 100
    top_n: int = 10
    exclusion_bars: int = 20
    nearby_gap_days: int = 20
    path_weight: float = 0.7
    algorithm: str = BASELINE_ALGORITHM
    load_start: str = "1900-01-01"


@dataclass(frozen=True)
class CrossSectionRequest:
    target_symbol: str
    universe_symbols: tuple[str, ...]
    start: str
    end: str
    top_n: int = 20
    min_coverage: float = 0.8
    path_weight: float = 0.7
    date_tolerance_bars: int = 0
    algorithm: str = BASELINE_ALGORITHM


@dataclass(frozen=True)
class ReviewRequest:
    symbol: str
    start: str
    end: str
    min_swing_return: float = 0.05
    min_segment_bars: int = 3
    max_segments: int = 6
    index_symbols: tuple[str, ...] = ("000300.SH", "000852.SH", "399006.SZ")
    proxy_symbols: tuple[str, ...] = ()
    industry_name: str = ""
    concept_name: str = ""
    sector_min_coverage: float = 0.5


@dataclass(frozen=True)
class DataCoverageRequest:
    symbols: tuple[str, ...]
    start: str
    end: str


@dataclass(frozen=True)
class DataUpdateRequest:
    symbols: tuple[str, ...]
    start: str
    end: str
    download_engine: str = "akshare"
    provider: str = ""
    trend_repo: str | Path | None = None


@dataclass(frozen=True)
class MissingDataUpdateRequest:
    symbols: tuple[str, ...]
    start: str
    end: str
    download_engine: str = "akshare"
    provider: str = ""
    trend_repo: str | Path | None = None


@dataclass(frozen=True)
class PriceImportRequest:
    file_path: str | Path
    fallback_symbol: str = ""


@dataclass(frozen=True)
class FullDailyUpdateRequest:
    start: str = "1990-01-01"
    end: str = pd.Timestamp.today().strftime("%Y-%m-%d")
    download_engine: str = "tdx"
    provider: str = ""
    batch_size: int = 100
    skip_available: bool = True
    include_indexes: bool = True
    extra_symbols: str = ""
    trend_repo: str | Path | None = None


@dataclass(frozen=True)
class QuickWindowResult:
    symbol: str
    start: str
    end: str
    rows: int
    message: str


@dataclass(frozen=True)
class FullDailyUpdatePlan:
    symbols: tuple[str, ...]
    download_symbols: tuple[str, ...]
    coverage: pd.DataFrame
    total_count: int
    stock_count: int
    index_count: int
    start: str
    end: str
    skip_available: bool


@dataclass(frozen=True)
class ReviewAnalysisBundle:
    reviews: list[ReviewResult]
    comparison_frames: list[tuple[str, pd.DataFrame]]
    comparison_frame: pd.DataFrame
    ranking_frame: pd.DataFrame
    script_profiles: list[dict[str, object]]
    warnings: tuple[str, ...]
    etf_matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_names: dict[str, str] = field(default_factory=dict)
    direction_by_symbol: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DataApiBarsResult:
    bars: pd.DataFrame
    download_status: pd.DataFrame | None = None
    meta: dict[str, object] | None = None


class DataApiClientProtocol(Protocol):
    def fetch_bars(
        self,
        *,
        symbols: tuple[str, ...],
        start: str,
        end: str,
        source: str,
        timeframe: str,
        adjust: str,
        tdx_path: str = "",
    ) -> DataApiBarsResult:
        ...


class DataApiError(RuntimeError):
    pass


class DataApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def fetch_bars(
        self,
        *,
        symbols: tuple[str, ...],
        start: str,
        end: str,
        source: str,
        timeframe: str,
        adjust: str,
        tdx_path: str = "",
    ) -> DataApiBarsResult:
        payload = self._get_json(
            "/api/v1/bars",
            {
                "symbols": ",".join(symbols),
                "start": start,
                "end": end,
                "source": source,
                "timeframe": timeframe,
                "adjust": adjust,
                "tdx_path": tdx_path,
            },
        )
        return DataApiBarsResult(
            bars=_records_to_bars(payload.get("records", [])),
            download_status=pd.DataFrame(payload.get("download_status", [])),
            meta=payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
        )

    def _get_json(self, path: str, params: dict[str, object]) -> dict[str, object]:
        query = urlencode({key: value for key, value in params.items() if str(value) != ""})
        url = f"{self.base_url}{path}?{query}"
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raise DataApiError(_http_error_message(exc)) from exc
        except URLError as exc:
            raise DataApiError(f"无法连接数据 API：{exc.reason}") from exc
        except TimeoutError as exc:
            raise DataApiError("数据 API 请求超时。") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DataApiError("数据 API 返回了非 JSON 响应。") from exc
        if not isinstance(payload, dict):
            raise DataApiError("数据 API 返回格式不是对象。")
        return payload


class DesktopSearchService:
    def __init__(
        self,
        config: DesktopAppConfig | None = None,
        *,
        data_api_client: DataApiClientProtocol | None = None,
    ) -> None:
        self.config = config or DesktopAppConfig()
        self.data_api_client = data_api_client

    def available_symbols(self) -> tuple[str, ...]:
        return tuple(available_symbols(self.config.data_root, self.config.timeframe, self.config.adjust))

    def load_bars(self, *, symbols: tuple[str, ...] | list[str], start: str, end: str) -> pd.DataFrame:
        return self._load_bars(symbols=symbols, start=start, end=end)

    def check_coverage(self, request: DataCoverageRequest) -> pd.DataFrame:
        symbols = parse_symbol_list(request.symbols)
        if not symbols:
            raise ValueError("至少需要一个检查代码。")
        return data_check(
            symbols=symbols,
            data_root=self.config.data_root,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            start=request.start,
            end=request.end,
        )

    def update_bars(self, request: DataUpdateRequest) -> pd.DataFrame:
        symbols = parse_symbol_list(request.symbols)
        if not symbols:
            raise ValueError("至少需要一个下载代码。")
        engine = str(request.download_engine or "akshare").strip().lower()
        provider = str(request.provider or "").strip()
        if engine == "tdx" and not provider:
            provider = str(self.config.tdx_path or "").strip()
        return update_local_bars(
            symbols=symbols,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            start=request.start,
            end=request.end,
            trend_repo=request.trend_repo,
            data_root=self.config.data_root,
            provider=provider,
            download_engine=engine,
        )

    def import_price_file(self, request: PriceImportRequest) -> pd.DataFrame:
        file_path = Path(request.file_path).expanduser()
        frame = read_price_data_file(file_path)
        return import_price_frame(
            data_root=self.config.data_root,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            frame=frame,
            fallback_symbol=request.fallback_symbol,
            source_name=file_path.name,
        )

    def update_missing_bars(self, request: MissingDataUpdateRequest) -> pd.DataFrame:
        symbols = parse_symbol_list(request.symbols)
        if not symbols:
            raise ValueError("至少需要一个检查代码。")
        checked = data_check(
            symbols=symbols,
            data_root=self.config.data_root,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            start=request.start,
            end=request.end,
        )
        download_symbols = tuple(_symbols_requiring_download(checked))
        if not download_symbols:
            return checked
        engine = str(request.download_engine or "akshare").strip().lower()
        provider = str(request.provider or "").strip()
        if engine == "tdx" and not provider:
            provider = str(self.config.tdx_path or "").strip()
        download_result = update_local_bars(
            symbols=download_symbols,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            start=request.start,
            end=request.end,
            trend_repo=request.trend_repo,
            data_root=self.config.data_root,
            provider=provider,
            download_engine=engine,
        )
        checked_after = data_check(
            symbols=download_symbols,
            data_root=self.config.data_root,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            start=request.start,
            end=request.end,
        )
        return _merge_download_check(download_result, checked_after)

    def plan_full_daily_update(self, request: FullDailyUpdateRequest) -> FullDailyUpdatePlan:
        stock_symbols = _full_daily_stock_symbols(request.download_engine, request.provider)
        symbols = tuple(
            symbols_with_analysis_indexes(
                stock_symbols,
                include_indexes=bool(request.include_indexes),
                extra_symbols=parse_symbol_list(request.extra_symbols),
            )
        )
        coverage = pd.DataFrame()
        download_symbols = symbols
        if request.skip_available:
            coverage = data_check(
                symbols=symbols,
                data_root=self.config.data_root,
                timeframe="1d",
                adjust=self.config.adjust,
                start=request.start,
                end=request.end,
            )
            download_symbols = tuple(_symbols_requiring_download(coverage))
        return FullDailyUpdatePlan(
            symbols=symbols,
            download_symbols=tuple(download_symbols),
            coverage=coverage,
            total_count=len(symbols),
            stock_count=len(unique_symbols(stock_symbols)),
            index_count=len(DEFAULT_ANALYSIS_INDEX_SYMBOLS) if request.include_indexes else 0,
            start=request.start,
            end=request.end,
            skip_available=bool(request.skip_available),
        )

    def run_full_daily_update(
        self,
        request: FullDailyUpdateRequest,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> pd.DataFrame:
        plan = self.plan_full_daily_update(request)
        if not plan.download_symbols:
            return plan.coverage.copy()
        frames: list[pd.DataFrame] = []
        batches = _batched(list(plan.download_symbols), max(1, int(request.batch_size)))
        completed = 0
        for batch_index, batch in enumerate(batches, start=1):
            download_result = update_local_bars(
                symbols=tuple(batch),
                timeframe="1d",
                adjust=self.config.adjust,
                start=request.start,
                end=request.end,
                trend_repo=request.trend_repo,
                data_root=self.config.data_root,
                provider=request.provider,
                download_engine=request.download_engine,
            )
            checked = data_check(
                symbols=tuple(batch),
                data_root=self.config.data_root,
                timeframe="1d",
                adjust=self.config.adjust,
                start=request.start,
                end=request.end,
            )
            frames.append(_merge_download_check(download_result, checked))
            completed += len(batch)
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed": completed,
                        "total": len(plan.download_symbols),
                        "batch_index": batch_index,
                        "batch_count": len(batches),
                        "current": ", ".join(batch),
                    }
                )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def resolve_review_auto_etf_proxies(
        self,
        *,
        industry_name: str,
        concept_name: str,
        etf_index: pd.DataFrame | None = None,
    ) -> tuple[tuple[str, ...], dict[str, str], pd.DataFrame, str]:
        queries = _etf_queries(industry_name, concept_name)
        index = _fallback_review_etf_index() if etf_index is None else etf_index
        matches = search_etf_index(index, queries)
        if queries and matches.empty:
            return (), {}, matches, f"ETF 名单没有匹配到：{', '.join(queries)}。"
        names = {
            normalize_symbol(row["symbol"]): str(row["name"]).strip()
            for _, row in matches.dropna(subset=["symbol"]).iterrows()
            if str(row.get("name", "")).strip()
        }
        return tuple(unique_symbols(matches.get("symbol", pd.Series(dtype=str)).dropna().astype(str).tolist())), names, matches, ""

    def popular_review_etfs(self, *, etf_index: pd.DataFrame | None = None, limit: int = 10) -> pd.DataFrame:
        index = _fallback_review_etf_index() if etf_index is None else etf_index
        return _top_etf_options(index, limit=limit)

    def quick_window(self, symbol: str, *, window_size: int, end: str | None = None) -> QuickWindowResult:
        symbols = parse_symbol_list(symbol)
        if not symbols:
            raise ValueError("至少需要一个目标代码。")
        if window_size < 1:
            raise ValueError("window_size 至少需要 1。")
        normalized = symbols[0]
        end_text = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        bars = self._load_bars(symbols=[normalized], start=DEFAULT_LOAD_START, end=end_text)
        dates = _valid_bar_dates(bars)
        if dates.empty:
            fallback = pd.Timestamp(end_text).strftime("%Y-%m-%d")
            return QuickWindowResult(
                symbol=normalized,
                start=fallback,
                end=fallback,
                rows=0,
                message=f"{normalized} 未找到有效行情日期，已按 {fallback} 设置区间。",
            )
        selected = dates.tail(window_size)
        start_text = pd.Timestamp(selected.iloc[0]).strftime("%Y-%m-%d")
        end_text = pd.Timestamp(selected.iloc[-1]).strftime("%Y-%m-%d")
        rows = int(len(selected))
        if rows < window_size:
            message = f"{normalized} 本地仅有 {rows} 根K线，不足近 {window_size} 根；已使用全部可用区间。"
        else:
            message = f"{normalized} 已选择近 {window_size} 根K线：{start_text} 至 {end_text}。"
        return QuickWindowResult(symbol=normalized, start=start_text, end=end_text, rows=rows, message=message)

    def latest_close(self, symbol: str) -> QuickWindowResult:
        symbols = parse_symbol_list(symbol)
        if not symbols:
            raise ValueError("至少需要一个目标代码。")
        normalized = symbols[0]
        bars = self._load_bars(
            symbols=[normalized],
            start=DEFAULT_LOAD_START,
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
        )
        dates = _valid_bar_dates(bars)
        if dates.empty:
            raise ValueError(f"{normalized} 未找到本地行情，无法设置最新收盘日。")
        latest_text = pd.Timestamp(dates.iloc[-1]).strftime("%Y-%m-%d")
        return QuickWindowResult(
            symbol=normalized,
            start=latest_text,
            end=latest_text,
            rows=int(len(dates)),
            message=f"{normalized} 已将区间结束设为最新本地收盘日：{latest_text}。",
        )

    def resolve_universe_symbols(
        self,
        *,
        symbols: str | tuple[str, ...] | list[str] = (),
        index_code: str = "",
        industry: str = "",
        concept: str = "",
    ) -> tuple[str, ...]:
        values: list[str] = list(parse_symbol_list(symbols))
        if index_code.strip():
            values.extend(universe_module.fetch_index_constituents(index_code.strip()))
        if industry.strip():
            values.extend(universe_module.fetch_industry_constituents(industry.strip()))
        if concept.strip():
            values.extend(universe_module.fetch_concept_constituents(concept.strip()))
        return tuple(unique_symbols(values))

    def plan_kline_migration(self, source: str | Path, destination: str | Path, *, overwrite: bool = False) -> pd.DataFrame:
        return plan_kline_migration(source, destination, overwrite=overwrite)

    def migrate_kline_data(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        mode: str = "copy",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        return migrate_kline_data(source, destination, mode=mode, overwrite=overwrite)

    def search_history(self, request: HistoryRequest) -> HistorySearchResult:
        symbol = normalize_symbol(request.symbol)
        if "," in str(request.symbol) or "\n" in str(request.symbol):
            symbols = parse_symbol_list(request.symbol)
            if len(symbols) != 1:
                raise ValueError("search_history 仅支持单个代码；多代码请调用 search_history_many。")
        bars = self._load_bars(
            symbols=[symbol],
            start=request.load_start,
            end=request.as_of,
        )
        return search_history(
            bars,
            HistorySearchConfig(
                symbol=symbol,
                as_of=request.as_of,
                window_size=int(request.window_size),
                window_start=request.window_start,
                forward_windows=tuple(int(item) for item in request.forward_windows),
                candidate_n=int(request.candidate_n),
                top_n=int(request.top_n),
                exclusion_bars=int(request.exclusion_bars),
                nearby_gap_days=int(request.nearby_gap_days),
                path_weight=float(request.path_weight),
                algorithm=request.algorithm,
            ),
        )

    def search_history_many(self, request: HistoryRequest) -> list[HistorySearchResult]:
        symbols = parse_symbol_list(request.symbol)
        if not symbols:
            raise ValueError("至少需要一个目标代码。")
        return [self.search_history(_history_request_for_symbol(request, symbol)) for symbol in symbols]

    def search_cross_section(self, request: CrossSectionRequest) -> CrossSectionSearchResult:
        target = normalize_symbol(request.target_symbol)
        if "," in str(request.target_symbol) or "\n" in str(request.target_symbol):
            targets = parse_symbol_list(request.target_symbol)
            if len(targets) != 1:
                raise ValueError("search_cross_section 仅支持单个目标代码；多目标请调用 search_cross_section_many。")
        universe = parse_symbol_list(request.universe_symbols)
        if not universe:
            universe = self.available_symbols()
        symbols = unique_symbols([target, *universe])
        bars = self._load_bars(
            symbols=symbols,
            start=_cross_section_load_start(request.start, request.date_tolerance_bars),
            end=_cross_section_load_end(request.end, request.date_tolerance_bars),
        )
        return search_cross_section(
            bars,
            CrossSectionSearchConfig(
                target_symbol=target,
                universe_symbols=tuple(universe),
                start=request.start,
                end=request.end,
                top_n=int(request.top_n),
                min_coverage=float(request.min_coverage),
                path_weight=float(request.path_weight),
                date_tolerance_bars=int(request.date_tolerance_bars),
                algorithm=request.algorithm,
            ),
        )

    def search_cross_section_many(self, request: CrossSectionRequest) -> list[CrossSectionSearchResult]:
        targets = parse_symbol_list(request.target_symbol)
        if not targets:
            raise ValueError("至少需要一个目标代码。")
        return [self.search_cross_section(_cross_section_request_for_target(request, target)) for target in targets]

    def cross_section_coverage(self, request: CrossSectionRequest) -> pd.DataFrame:
        targets = parse_symbol_list(request.target_symbol)
        if not targets:
            raise ValueError("至少需要一个目标代码。")
        universe = parse_symbol_list(request.universe_symbols)
        if not universe:
            universe = self.available_symbols()
        return data_check(
            symbols=tuple(unique_symbols([*targets, *universe])),
            data_root=self.config.data_root,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            start=_cross_section_load_start(request.start, request.date_tolerance_bars),
            end=_cross_section_load_end(request.end, request.date_tolerance_bars),
        )

    def analyze_review(self, request: ReviewRequest) -> ReviewResult:
        symbol = normalize_symbol(request.symbol)
        if "," in str(request.symbol) or "\n" in str(request.symbol):
            symbols = parse_symbol_list(request.symbol)
            if len(symbols) != 1:
                raise ValueError("analyze_review 仅支持单个代码；多代码请调用 analyze_review_many。")
        bars = self._load_bars(
            symbols=[symbol],
            start=request.start,
            end=request.end,
        )
        return analyze_price_review(
            bars,
            ReviewConfig(
                symbol=symbol,
                start=request.start,
                end=request.end,
                min_swing_return=float(request.min_swing_return),
                min_segment_bars=int(request.min_segment_bars),
                max_segments=int(request.max_segments),
            ),
        )

    def analyze_review_many(self, request: ReviewRequest) -> list[ReviewResult]:
        symbols = parse_symbol_list(request.symbol)
        if not symbols:
            raise ValueError("至少需要一个目标代码。")
        bars = self._load_bars(symbols=symbols, start=request.start, end=request.end)
        return [
            analyze_price_review(
                bars.loc[bars["stock_code"] == symbol],
                ReviewConfig(
                    symbol=symbol,
                    start=request.start,
                    end=request.end,
                    min_swing_return=float(request.min_swing_return),
                    min_segment_bars=int(request.min_segment_bars),
                    max_segments=int(request.max_segments),
                ),
            )
            for symbol in symbols
        ]

    def analyze_review_bundle(self, request: ReviewRequest) -> ReviewAnalysisBundle:
        symbols = parse_symbol_list(request.symbol)
        if not symbols:
            raise ValueError("至少需要一个目标代码。")
        index_symbols = parse_symbol_list(request.index_symbols)
        auto_proxy_symbols, auto_proxy_names, etf_matches, etf_warning = self.resolve_review_auto_etf_proxies(
            industry_name=request.industry_name,
            concept_name=request.concept_name,
        )
        proxy_symbols = tuple(unique_symbols([*parse_symbol_list(request.proxy_symbols), *auto_proxy_symbols]))
        sector_symbols, sector_warnings = _review_sector_symbols(
            request.industry_name,
            request.concept_name,
        )
        direct_symbols = tuple(
            unique_symbols([*symbols, *index_symbols, *proxy_symbols, SCRIPT_BENCHMARK_SYMBOL, *sector_symbols])
        )
        bars = self._load_bars(
            symbols=direct_symbols,
            start=_review_data_start(request.start, request.end),
            end=request.end,
        )
        stock_names = {
            **_stock_names_from_akshare(direct_symbols),
            **_stock_names_from_bars(bars, direct_symbols),
            **auto_proxy_names,
        }
        direction_label = _review_direction_label(request.industry_name, request.concept_name)
        direction_by_symbol = {symbol: direction_label for symbol in symbols}
        reviews = [
            analyze_price_review(
                bars.loc[bars["stock_code"] == symbol],
                ReviewConfig(
                    symbol=symbol,
                    start=request.start,
                    end=request.end,
                    min_swing_return=float(request.min_swing_return),
                    min_segment_bars=int(request.min_segment_bars),
                    max_segments=int(request.max_segments),
                ),
            )
            for symbol in symbols
        ]
        valid_reviews = [result for result in reviews if not result.window.empty]
        comparison_frames, comparison_warnings = _review_comparison_frames(
            bars,
            index_symbols=index_symbols,
            proxy_symbols=proxy_symbols,
            sector_symbols=sector_symbols,
            industry_name=request.industry_name,
            concept_name=request.concept_name,
            sector_min_coverage=float(request.sector_min_coverage),
            start=request.start,
            end=request.end,
        )
        comparison_frame = pd.DataFrame(_review_comparison_rows(valid_reviews, comparison_frames))
        ranking_frame = rank_review_results(
            valid_reviews,
            comparison_frame,
            stock_names=stock_names,
            direction_by_symbol=direction_by_symbol,
        )
        if not ranking_frame.empty and auto_proxy_names:
            ranking_frame["股票"] = ranking_frame.apply(
                lambda row: str(row.get("股票", "") or auto_proxy_names.get(str(row.get("代码", "")), "")),
                axis=1,
            )
        ranking_records = ranking_frame.set_index("代码").to_dict("index") if not ranking_frame.empty else {}
        benchmark = _symbol_window(bars, SCRIPT_BENCHMARK_SYMBOL, request.start, request.end)
        script_profiles: list[dict[str, object]] = []
        for result in valid_reviews:
            profile = build_video_script_profile(
                result,
                _symbol_window(bars, result.symbol, _review_data_start(request.start, request.end), request.end),
                benchmark if not benchmark.empty else None,
                benchmark_label=SCRIPT_BENCHMARK_SYMBOL,
            )
            profile.update(
                {
                    key: value
                    for key, value in ranking_records.get(result.symbol, {}).items()
                    if key
                    in {
                        "排名",
                        "股票",
                        "所属方向",
                        "对标指数",
                        "指数阶段",
                        "强弱等级",
                        "区间收益",
                        "最大回撤",
                        "相对超额",
                        "关键转折点",
                        "当前性质",
                        "锐评结论",
                        "明日验证",
                    }
                }
            )
            script_profiles.append(profile)
        warnings = tuple(
            dict.fromkeys(
                [
                    *sector_warnings,
                    *comparison_warnings,
                    *([etf_warning] if etf_warning else []),
                    *(warning for item in reviews for warning in item.warnings),
                ]
            )
        )
        return ReviewAnalysisBundle(
            reviews=reviews,
            comparison_frames=comparison_frames,
            comparison_frame=comparison_frame,
            ranking_frame=ranking_frame,
            script_profiles=script_profiles,
            warnings=warnings,
            etf_matches=etf_matches,
            stock_names=stock_names,
            direction_by_symbol=direction_by_symbol,
        )

    def _load_bars(self, *, symbols: tuple[str, ...] | list[str], start: str, end: str) -> pd.DataFrame:
        mode = self.config.data_mode.strip().lower()
        if mode not in DATA_MODES:
            raise ValueError("data_mode 仅支持 local 或 api。")
        source = self.config.data_source.strip().lower()
        if source not in DATA_SOURCES:
            raise ValueError("data_source 仅支持 local、akshare 或 tdx。")
        normalized_symbols = tuple(unique_symbols(symbols))
        if mode == "api":
            client = self.data_api_client or DataApiClient(self.config.api_base_url)
            return client.fetch_bars(
                symbols=normalized_symbols,
                start=start,
                end=end,
                source=source,
                timeframe=self.config.timeframe,
                adjust=self.config.adjust,
                tdx_path=self.config.tdx_path,
            ).bars
        return load_local_bars(
            data_root=self.config.data_root,
            timeframe=self.config.timeframe,
            adjust=self.config.adjust,
            symbols=normalized_symbols,
            start=start,
            end=end,
        )


def _full_daily_stock_symbols(download_engine: str, provider: str) -> list[str]:
    engine = str(download_engine or "tdx").strip().lower()
    if engine == "tdx":
        return fetch_tdx_stock_symbols(tqcenter_path=provider)
    return fetch_all_a_symbols()


def _symbols_requiring_download(check: pd.DataFrame) -> list[str]:
    if check.empty or not {"symbol", "status"}.issubset(check.columns):
        return []
    missing = check.loc[check["status"].astype(str).isin(DOWNLOAD_REQUIRED_STATUSES), "symbol"]
    return unique_symbols(missing.astype(str).tolist())


def _merge_download_check(download_result: pd.DataFrame, checked: pd.DataFrame) -> pd.DataFrame:
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
        if str(download_row.get("status", "")) == "failed":
            result.loc[index, "status"] = "failed"
            result.loc[index, "message"] = str(download_row.get("message", ""))
        elif row["status"] != "available":
            message = str(row.get("message", ""))
            result.loc[index, "message"] = f"下载后仍未覆盖：{message}".rstrip("：")
    return result


def _batched(symbols: list[str], batch_size: int) -> list[list[str]]:
    size = max(1, int(batch_size))
    return [symbols[start : start + size] for start in range(0, len(symbols), size)]


def _fallback_review_etf_index() -> pd.DataFrame:
    return pd.DataFrame(FALLBACK_REVIEW_ETFS, columns=["symbol", "name", "amount", "category"])


def search_etf_index(
    etf_index: pd.DataFrame,
    queries: list[str] | tuple[str, ...],
    *,
    limit_per_query: int = 1,
) -> pd.DataFrame:
    columns = ["query", "symbol", "name", "amount", "category"]
    if etf_index.empty:
        return pd.DataFrame(columns=columns)
    frame = etf_index.copy()
    for column in ["symbol", "name", "category"]:
        if column not in frame.columns:
            frame[column] = ""
    if "amount" not in frame.columns:
        frame["amount"] = 0.0
    frame["symbol"] = frame["symbol"].map(normalize_symbol)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    rows: list[pd.DataFrame] = []
    for raw_query in queries:
        query = _etf_query_key(raw_query)
        if not query:
            continue
        mask = (
            frame["name"].astype(str).map(_etf_query_key).str.contains(query, regex=False, na=False)
            | frame["category"].astype(str).map(_etf_query_key).str.contains(query, regex=False, na=False)
        )
        matched = (
            frame.loc[mask]
            .sort_values(["amount", "symbol"], ascending=[False, True])
            .head(max(1, int(limit_per_query)))
            .copy()
        )
        if matched.empty:
            continue
        matched.insert(0, "query", str(raw_query).strip())
        rows.append(matched[columns])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).drop_duplicates("symbol").reset_index(drop=True)


def _top_etf_options(index: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    columns = ["symbol", "name", "amount", "category"]
    if index.empty:
        return pd.DataFrame(columns=columns)
    result = index.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = "" if column != "amount" else 0.0
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["name"] = result["name"].fillna("").astype(str).str.strip()
    result["category"] = result["category"].fillna("").astype(str).str.strip()
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce").fillna(0.0)
    result = result.loc[result["symbol"].ne("") & result["name"].ne("")]
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["_theme"] = result["category"].where(result["category"].ne(""), result["symbol"])
    result = (
        result.sort_values(["amount", "symbol"], ascending=[False, True])
        .drop_duplicates("_theme", keep="first")
        .head(max(0, int(limit)))
    )
    return result[columns].reset_index(drop=True)


def _etf_queries(industry_name: str, concept_name: str) -> list[str]:
    return [text for text in dict.fromkeys([str(industry_name or "").strip(), str(concept_name or "").strip()]) if text]


def _etf_query_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return re.sub(r"[\s　（）()【】\[\]：:·•,，、;；/\\-]+", "", text)


def _review_data_start(start: str, end: str) -> str:
    start_ts = pd.Timestamp(start)
    ytd_start = pd.Timestamp(year=pd.Timestamp(end).year, month=1, day=1)
    context_start = start_ts - pd.Timedelta(days=90)
    return min(context_start, ytd_start).strftime("%Y-%m-%d")


def _review_sector_symbols(industry_name: str, concept_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    symbols: list[str] = []
    warnings: list[str] = []
    industry = str(industry_name or "").strip()
    concept = str(concept_name or "").strip()
    if industry:
        try:
            symbols.extend(universe_module.fetch_industry_constituents(industry))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"行业板块 {industry} 成分获取失败：{exc}")
    if concept:
        try:
            symbols.extend(universe_module.fetch_concept_constituents(concept))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"概念板块 {concept} 成分获取失败：{exc}")
    return tuple(unique_symbols(symbols)), tuple(warnings)


def _review_comparison_frames(
    bars: pd.DataFrame,
    *,
    index_symbols: tuple[str, ...],
    proxy_symbols: tuple[str, ...],
    sector_symbols: tuple[str, ...],
    industry_name: str,
    concept_name: str,
    sector_min_coverage: float,
    start: str,
    end: str,
) -> tuple[list[tuple[str, pd.DataFrame]], tuple[str, ...]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    warnings: list[str] = []
    for symbol in unique_symbols([*index_symbols, *proxy_symbols]):
        frame = _symbol_window(bars, symbol, start, end)
        if frame.empty:
            warnings.append(f"{symbol} 缺少本地行情，未纳入对比。")
            continue
        frames.append((symbol, frame))
    sector_label = _sector_label(industry_name, concept_name)
    if sector_symbols and sector_label:
        sector_bars = bars.loc[bars["stock_code"].isin(sector_symbols)].copy()
        equal_weight = build_equal_weight_series(
            _filter_date_range(sector_bars, start, end),
            sector_symbols,
            label=sector_label,
            min_coverage=sector_min_coverage,
        )
        if equal_weight.warning:
            warnings.append(equal_weight.warning)
        if not equal_weight.frame.empty:
            frames.append((sector_label, equal_weight.frame))
    return frames, tuple(warnings)


def _review_comparison_rows(
    results: list[ReviewResult] | tuple[ReviewResult, ...],
    comparison_frames: list[tuple[str, pd.DataFrame]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for label, frame in comparison_frames:
            row = build_comparison_stats(result.window, frame, label)
            row["代码"] = result.symbol
            rows.append(row)
    return rows


def _symbol_window(bars: pd.DataFrame, symbol: str, start: str, end: str) -> pd.DataFrame:
    if bars.empty or "stock_code" not in bars.columns:
        return pd.DataFrame()
    normalized = normalize_symbol(symbol)
    frame = bars.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    return _filter_date_range(frame.loc[frame["stock_code"] == normalized], start, end)


def _filter_date_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame.copy()
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    mask = result["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    return result.loc[mask].sort_values("date").reset_index(drop=True)


def _sector_label(industry_name: str, concept_name: str) -> str:
    industry = str(industry_name or "").strip()
    concept = str(concept_name or "").strip()
    if industry and concept:
        return f"行业:{industry} / 概念:{concept}"
    if industry:
        return f"行业:{industry}"
    if concept:
        return f"概念:{concept}"
    return ""


def _review_direction_label(industry_name: str, concept_name: str) -> str:
    return _sector_label(industry_name, concept_name) or "自选复盘"


def _stock_names_from_akshare(symbols: tuple[str, ...] | list[str]) -> dict[str, str]:
    normalized = tuple(unique_symbols(symbols))
    if not normalized:
        return {}
    tables: list[pd.DataFrame] = []
    try:
        import akshare as ak
    except Exception:  # noqa: BLE001
        return {}
    for func_name in ("stock_info_a_code_name", "stock_zh_a_spot_em"):
        fetcher = getattr(ak, func_name, None)
        if not callable(fetcher):
            continue
        try:
            tables.append(fetcher())
        except Exception:  # noqa: BLE001
            continue
    names: dict[str, str] = {}
    for table in tables:
        names.update(_stock_name_map_from_table(table, normalized))
    return names


def _stock_names_from_bars(bars: pd.DataFrame, symbols: tuple[str, ...] | list[str]) -> dict[str, str]:
    if bars.empty:
        return {}
    return _stock_name_map_from_table(bars, tuple(unique_symbols(symbols)))


def _stock_name_map_from_table(table: pd.DataFrame, symbols: tuple[str, ...]) -> dict[str, str]:
    if table.empty:
        return {}
    code_column = next(
        (column for column in ("code", "stock_code", "symbol", "ts_code", "证券代码", "代码", "股票代码") if column in table.columns),
        "",
    )
    name_column = next(
        (column for column in ("name", "stock_name", "股票名称", "证券简称", "名称", "简称") if column in table.columns),
        "",
    )
    if not code_column or not name_column:
        return {}
    wanted = set(unique_symbols(symbols))
    names: dict[str, str] = {}
    for _, row in table[[code_column, name_column]].dropna(subset=[code_column]).iterrows():
        symbol = normalize_symbol(row[code_column])
        if symbol in wanted:
            name = "" if pd.isna(row[name_column]) else str(row[name_column]).strip()
            if name:
                names[symbol] = name
    return names


def _cross_section_load_start(start: str | pd.Timestamp, date_tolerance_bars: int) -> str:
    return (pd.Timestamp(start) - pd.Timedelta(days=_date_tolerance_calendar_days(date_tolerance_bars))).strftime(
        "%Y-%m-%d"
    )


def _cross_section_load_end(end: str | pd.Timestamp, date_tolerance_bars: int) -> str:
    requested_end = pd.Timestamp(end) + pd.Timedelta(days=_date_tolerance_calendar_days(date_tolerance_bars))
    today = pd.Timestamp.today().normalize()
    if requested_end >= today:
        return requested_end.strftime("%Y-%m-%d")
    return min(requested_end + pd.Timedelta(days=45), today).strftime("%Y-%m-%d")


def _date_tolerance_calendar_days(date_tolerance_bars: int) -> int:
    if date_tolerance_bars <= 0:
        return 0
    return max(date_tolerance_bars + 2, int(date_tolerance_bars * 2.2 + 0.999999))


def _history_request_for_symbol(request: HistoryRequest, symbol: str) -> HistoryRequest:
    return HistoryRequest(
        symbol=symbol,
        as_of=request.as_of,
        window_size=request.window_size,
        window_start=request.window_start,
        forward_windows=request.forward_windows,
        candidate_n=request.candidate_n,
        top_n=request.top_n,
        exclusion_bars=request.exclusion_bars,
        nearby_gap_days=request.nearby_gap_days,
        path_weight=request.path_weight,
        algorithm=request.algorithm,
        load_start=request.load_start,
    )


def _cross_section_request_for_target(request: CrossSectionRequest, target: str) -> CrossSectionRequest:
    return CrossSectionRequest(
        target_symbol=target,
        universe_symbols=tuple(symbol for symbol in request.universe_symbols if normalize_symbol(symbol) != target),
        start=request.start,
        end=request.end,
        top_n=request.top_n,
        min_coverage=request.min_coverage,
        path_weight=request.path_weight,
        date_tolerance_bars=request.date_tolerance_bars,
        algorithm=request.algorithm,
    )


def _records_to_bars(records: object) -> pd.DataFrame:
    frame = pd.DataFrame(records if isinstance(records, list) else [])
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"])
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _http_error_message(exc: HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return f"数据 API 请求失败：HTTP {exc.code}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str):
        return detail
    return f"数据 API 请求失败：HTTP {exc.code}"
