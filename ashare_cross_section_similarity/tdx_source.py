from __future__ import annotations

from collections.abc import Mapping
import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.data import CANONICAL_COLUMNS
from ashare_cross_section_similarity.universe import SYMBOL_COLUMNS, normalize_symbol, symbols_from_table, unique_symbols

TDX_TQCENTER_ENV_VAR = "TDX_TQCENTER_PATH"
DEFAULT_TDX_TQCENTER_PATH = r"F:\new_tdx64\PYPlugins"
TDX_REQUEST_BATCH_SIZE = 100
TIMEFRAME_PERIODS = {"1d": "1d", "30m": "30m", "15m": "15m", "5m": "5m", "1m": "1m"}
ADJUST_MAP = {"": "none", "qfq": "front", "hfq": "back"}
REQUIRED_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount")
FIELD_ALIASES = {
    "Open": ("Open", "open"),
    "High": ("High", "high"),
    "Low": ("Low", "low"),
    "Close": ("Close", "close"),
    "Volume": ("Volume", "volume", "vol"),
    "Amount": ("Amount", "amount"),
}
OUTPUT_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
}
STOCK_LIST_METHODS = (
    "get_stock_list",
    "get_security_list",
    "get_code_list",
    "get_instrument_list",
    "get_instrument_detail",
)
STOCK_LIST_MARKETS = ("SH", "SZ", "BJ", 1, 0, 2)
A_SHARE_STOCK_PREFIXES = {
    "SH": ("600", "601", "603", "605", "688", "689"),
    "SZ": ("000", "001", "002", "003", "300", "301"),
    "BJ": ("4", "8", "920"),
}
TDX_MARKET_COLUMNS = ("market", "exchange", "mkt", "市场", "交易所", "交易市场")
TDX_NAME_COLUMNS = ("name", "stock_name", "security_name", "证券简称", "证券名称", "名称", "简称", "股票名称")
TDX_AMOUNT_COLUMNS = ("amount", "Amount", "turnover", "成交额", "成交金额", "成交额(元)", "金额")
TDX_BLOCK_INDEX_PREFIXES = ("880", "881", "882", "883", "884", "885", "886", "887", "888", "889")
TDX_KLINE_SYMBOL_TABLE_COLUMNS = ["symbol", "name", "category"]
TDX_KLINE_CATEGORY_ORDER = {"stock": 0, "etf": 1, "index": 2, "other": 3}
TDX_KLINE_PREFIXES = {
    "SH": (*A_SHARE_STOCK_PREFIXES["SH"], "000", "5", *TDX_BLOCK_INDEX_PREFIXES),
    "SZ": (*A_SHARE_STOCK_PREFIXES["SZ"], "15", "16", "18", "399"),
    "BJ": A_SHARE_STOCK_PREFIXES["BJ"],
}

_TQ_CLIENT: Any | None = None
_TQ_CLIENT_IMPORT_KEY: str | None = None
_TQ_CLIENT_SYS_PATHS: set[str] = set()
_INITIALIZED = False
_INITIALIZED_CLIENT_ID: int | None = None


def fetch_tdx_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    timeframe: str = "1d",
    adjust: str = "qfq",
    tqcenter_path: str = "",
    tq_client: Any | None = None,
) -> pd.DataFrame:
    period = TIMEFRAME_PERIODS.get(timeframe)
    if period is None:
        raise ValueError("timeframe 仅支持 1d、30m、15m、5m、1m。")
    dividend_type = ADJUST_MAP.get(adjust)
    if dividend_type is None:
        raise ValueError("adjust 仅支持 qfq、hfq 或空字符串。")

    tq = tq_client or _load_tq(tqcenter_path)
    _ensure_initialized(tq)
    normalized_symbols = unique_symbols(symbols)
    if not normalized_symbols:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    frames: list[pd.DataFrame] = []
    for symbol_batch in _batched_symbols(normalized_symbols, TDX_REQUEST_BATCH_SIZE):
        payload = tq.get_market_data(
            field_list=list(REQUIRED_FIELDS),
            stock_list=symbol_batch,
            period=period,
            start_time=_format_market_time(start),
            end_time=_format_market_time(end),
            count=-1,
            dividend_type=dividend_type,
            fill_data=False,
        )
        allow_missing_symbol = len(symbol_batch) > 1
        for symbol in symbol_batch:
            frame = _normalize_tdx_payload(
                payload,
                symbol=symbol,
                start=start,
                end=end,
                allow_missing_symbol=allow_missing_symbol,
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def fetch_tdx_stock_symbols(*, tqcenter_path: str = "", tq_client: Any | None = None) -> list[str]:
    tq = tq_client or _load_tq(tqcenter_path)

    symbols, errors = _collect_tdx_stock_symbols(tq)
    if symbols:
        return symbols
    if _tdx_errors_need_initialize(errors):
        _ensure_initialized(tq)
        symbols, retry_errors = _collect_tdx_stock_symbols(tq)
        if symbols:
            return symbols
        errors = [*errors, "初始化后重试仍失败", *retry_errors]
    details = " | ".join(errors)
    raise RuntimeError(f"TDX 未能获取股票清单。请确认 tqcenter 支持股票列表接口。详情: {details}")


def _collect_tdx_stock_symbols(tq: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    for method_name in STOCK_LIST_METHODS:
        method = getattr(tq, method_name, None)
        if method is None:
            errors.append(f"{method_name}: unavailable")
            continue

        method_symbols: list[str] = []
        for label, args, kwargs in _stock_list_call_variants():
            try:
                payload = method(*args, **kwargs)
            except TypeError as exc:
                errors.append(f"{method_name}{label}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method_name}{label}: {exc}")
                continue

            symbols = _symbols_from_tdx_stock_payload(payload)
            if symbols:
                method_symbols.extend(symbols)
                if label == "()":
                    break

        if method_symbols:
            stock_symbols = _filter_a_share_stock_symbols(method_symbols)
            if stock_symbols:
                return stock_symbols, errors
            errors.append(f"{method_name}: 返回结果未包含 A 股股票代码")

    return [], errors


def fetch_tdx_kline_symbols(*, tqcenter_path: str = "", tq_client: Any | None = None) -> list[str]:
    return fetch_tdx_kline_symbol_table(tqcenter_path=tqcenter_path, tq_client=tq_client)["symbol"].tolist()


def fetch_tdx_kline_symbol_table(*, tqcenter_path: str = "", tq_client: Any | None = None) -> pd.DataFrame:
    tq = tq_client or _load_tq(tqcenter_path)

    table, errors = _collect_tdx_kline_symbol_table(tq)
    if not table.empty:
        return table
    if _tdx_errors_need_initialize(errors):
        _ensure_initialized(tq)
        table, retry_errors = _collect_tdx_kline_symbol_table(tq)
        if not table.empty:
            return table
        errors = [*errors, "初始化后重试仍失败", *retry_errors]
    details = " | ".join(errors)
    raise RuntimeError(f"TDX 未能获取 K 线标的清单。请确认 tqcenter 支持股票列表接口。详情: {details}")


def _collect_tdx_kline_symbol_table(tq: Any) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    for method_name in STOCK_LIST_METHODS:
        method = getattr(tq, method_name, None)
        if method is None:
            errors.append(f"{method_name}: unavailable")
            continue

        method_tables: list[pd.DataFrame] = []
        for label, args, kwargs in _stock_list_call_variants():
            try:
                payload = method(*args, **kwargs)
            except TypeError as exc:
                errors.append(f"{method_name}{label}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method_name}{label}: {exc}")
                continue

            table = _tdx_kline_symbol_table_from_payload(payload)
            if not table.empty:
                method_tables.append(table)

        if method_tables:
            kline_table = _deduplicate_tdx_kline_symbol_table(pd.concat(method_tables, ignore_index=True))
            if not kline_table.empty:
                return kline_table, errors
            errors.append(f"{method_name}: 返回结果未包含可下载日 K 的股票、ETF 或板块指数代码")

    return _empty_tdx_kline_symbol_table(), errors


def fetch_tdx_etf_index(*, tqcenter_path: str = "", tq_client: Any | None = None) -> pd.DataFrame:
    tq = tq_client or _load_tq(tqcenter_path)

    table, errors = _collect_tdx_etf_index(tq)
    if not table.empty:
        return table
    if _tdx_errors_need_initialize(errors):
        _ensure_initialized(tq)
        table, retry_errors = _collect_tdx_etf_index(tq)
        if not table.empty:
            return table
        errors = [*errors, "初始化后重试仍失败", *retry_errors]
    if errors and all("unavailable" in error for error in errors):
        details = " | ".join(errors)
        raise RuntimeError(f"TDX 未能获取 ETF 清单。请确认 tqcenter 支持股票列表接口。详情: {details}")
    return _empty_etf_index()


def _collect_tdx_etf_index(tq: Any) -> tuple[pd.DataFrame, list[str]]:
    tables: list[pd.DataFrame] = []
    errors: list[str] = []
    for method_name in STOCK_LIST_METHODS:
        method = getattr(tq, method_name, None)
        if method is None:
            errors.append(f"{method_name}: unavailable")
            continue
        for label, args, kwargs in _stock_list_call_variants():
            try:
                payload = method(*args, **kwargs)
            except TypeError as exc:
                errors.append(f"{method_name}{label}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{method_name}{label}: {exc}")
                continue
            table = build_tdx_etf_index(payload)
            if not table.empty:
                tables.append(table)

    if tables:
        return _normalize_etf_index(pd.concat(tables, ignore_index=True)), errors
    return _empty_etf_index(), errors


def build_tdx_etf_index(payload: Any) -> pd.DataFrame:
    tables = _tdx_list_tables_from_payload(payload)
    rows: list[dict[str, object]] = []
    for table in tables:
        if table.empty:
            continue
        code_column = next((column for column in SYMBOL_COLUMNS if column in table.columns), None)
        name_column = next((column for column in TDX_NAME_COLUMNS if column in table.columns), None)
        if code_column is None or name_column is None:
            continue
        market_column = next((column for column in TDX_MARKET_COLUMNS if column in table.columns), None)
        amount_column = next((column for column in TDX_AMOUNT_COLUMNS if column in table.columns), None)
        for _, row in table.iterrows():
            name = str(row.get(name_column, "")).strip()
            symbol = _normalize_tdx_list_symbol(row.get(code_column), market_hint=row.get(market_column) if market_column else None)
            if not _is_tdx_etf(symbol, name):
                continue
            amount = _parse_tdx_amount(row.get(amount_column)) if amount_column else 0.0
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "amount": amount,
                    "category": _etf_category_key(name),
                }
            )
    if not rows:
        return _empty_etf_index()
    return _normalize_etf_index(pd.DataFrame(rows))


def search_tdx_etf_index(
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
        matched = frame.loc[mask].sort_values(["amount", "symbol"], ascending=[False, True]).head(max(1, int(limit_per_query))).copy()
        if matched.empty:
            continue
        matched.insert(0, "query", str(raw_query).strip())
        rows.append(matched[columns])
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["query", "amount"], ascending=[True, False]).drop_duplicates("symbol").reset_index(drop=True)


def _load_tq(tqcenter_path: str = "") -> Any:
    global _INITIALIZED, _INITIALIZED_CLIENT_ID, _TQ_CLIENT, _TQ_CLIENT_IMPORT_KEY
    candidate_paths = _candidate_import_paths(tqcenter_path)
    cache_key = _tq_import_cache_key(candidate_paths)
    if _TQ_CLIENT is not None and _TQ_CLIENT_IMPORT_KEY == cache_key:
        return _TQ_CLIENT
    if _TQ_CLIENT is not None:
        _TQ_CLIENT = None
        _TQ_CLIENT_IMPORT_KEY = None
        _INITIALIZED = False
        _INITIALIZED_CLIENT_ID = None
    if candidate_paths:
        _remove_tqcenter_import_paths()
        sys.modules.pop("tqcenter", None)

    errors: list[str] = []
    for path in candidate_paths:
        resolved = path.resolve()
        if not resolved.exists():
            errors.append(f"{resolved} 不存在")
            continue
        inserted = False
        path_text = str(resolved)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            _TQ_CLIENT_SYS_PATHS.add(path_text)
            inserted = True
        try:
            module = importlib.import_module("tqcenter")
            _TQ_CLIENT = getattr(module, "tq")
            _TQ_CLIENT_IMPORT_KEY = cache_key
            return _TQ_CLIENT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_text}: {exc}")
            if inserted:
                try:
                    sys.path.remove(path_text)
                except ValueError:
                    pass
                _TQ_CLIENT_SYS_PATHS.discard(path_text)

    try:
        _remove_tqcenter_import_paths()
        module = importlib.import_module("tqcenter")
        _TQ_CLIENT = getattr(module, "tq")
        _TQ_CLIENT_IMPORT_KEY = cache_key
        return _TQ_CLIENT
    except Exception as exc:  # noqa: BLE001
        errors.append(f"normal import: {exc}")
        details = " | ".join(errors)
        raise RuntimeError(
            "无法导入 tqcenter。请先安装并登录本机通达信终端，并通过"
            f" {TDX_TQCENTER_ENV_VAR} 或下载源输入框指向 TDX 的 PYPlugins/user 目录。"
            f" 详情: {details}"
        ) from exc


def _tdx_errors_need_initialize(errors: list[str]) -> bool:
    return any("TQ数据接口初始化失败" in error or "初始化失败" in error for error in errors)


def _remove_tqcenter_import_paths() -> None:
    for path_text in list(_TQ_CLIENT_SYS_PATHS):
        while path_text in sys.path:
            sys.path.remove(path_text)
        _TQ_CLIENT_SYS_PATHS.discard(path_text)


def _tq_import_cache_key(paths: list[Path]) -> str:
    if paths:
        return os.pathsep.join(str(path.expanduser().resolve()) for path in paths)
    return "normal-import"


def _candidate_import_paths(tqcenter_path: str = "") -> list[Path]:
    raw_value = (tqcenter_path or os.getenv(TDX_TQCENTER_ENV_VAR, "") or DEFAULT_TDX_TQCENTER_PATH).strip()
    if not raw_value or raw_value.lower() == "tdx":
        return []

    def expand(candidate: Path) -> list[Path]:
        normalized = candidate
        if normalized.name.lower() == "tqcenter.py":
            normalized = normalized.parent
        if normalized.name.lower() == "user" and normalized.parent.name.lower() == "pyplugins":
            return [normalized]
        if normalized.name.lower() == "pyplugins":
            return [normalized / "user", normalized]
        return [normalized / "PYPlugins" / "user", normalized]

    paths: list[Path] = []
    seen: set[str] = set()
    for item in _split_tqcenter_path_items(raw_value):
        text = _normalize_tqcenter_path_text(item)
        if not text:
            continue
        for path in expand(Path(text).expanduser()):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def _split_tqcenter_path_items(raw_value: str) -> list[str]:
    if os.pathsep != ":":
        return raw_value.split(os.pathsep)
    items: list[str] = []
    current: list[str] = []
    for index, character in enumerate(raw_value):
        if character == os.pathsep and not _is_windows_drive_separator(raw_value, index):
            items.append("".join(current))
            current = []
            continue
        current.append(character)
    items.append("".join(current))
    return items


def _is_windows_drive_separator(text: str, index: int) -> bool:
    return index == 1 and len(text) > 2 and text[0].isalpha() and text[2] in {"\\", "/"}


def _normalize_tqcenter_path_text(text: str) -> str:
    stripped = text.strip().strip('"')
    if re.match(r"^[A-Za-z]:[\\/]", stripped):
        return stripped.replace("\\", "/")
    return stripped


def _stock_list_call_variants() -> list[tuple[str, tuple[object, ...], dict[str, object]]]:
    variants: list[tuple[str, tuple[object, ...], dict[str, object]]] = [("()", (), {})]
    for market in STOCK_LIST_MARKETS:
        variants.append((f"({market!r})", (market,), {}))
        variants.append((f"(market={market!r})", (), {"market": market}))
    return variants


def _symbols_from_tdx_stock_payload(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, pd.DataFrame):
        return _symbols_from_stock_table(payload)
    if isinstance(payload, pd.Series):
        return unique_symbols(payload.dropna().tolist())
    if isinstance(payload, str):
        return unique_symbols([payload])
    if isinstance(payload, Mapping):
        symbols = _symbols_from_mapping_values(payload)
        if symbols:
            return symbols
        try:
            return _symbols_from_stock_table(pd.DataFrame(payload))
        except Exception:  # noqa: BLE001
            return []
    if isinstance(payload, (list, tuple, set)):
        values = list(payload)
        if not values:
            return []
        if all(isinstance(item, Mapping) for item in values):
            return _symbols_from_stock_table(pd.DataFrame(values))
        if all(isinstance(item, (str, int)) for item in values):
            return unique_symbols(values)
        symbols: list[str] = []
        for item in values:
            symbols.extend(_symbols_from_tdx_stock_payload(item))
        return unique_symbols(symbols)
    return []


def _symbols_from_tdx_kline_payload(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, pd.DataFrame):
        return _symbols_from_tdx_kline_table(payload)
    if isinstance(payload, pd.Series):
        return _unique_tdx_symbols(payload.dropna().tolist())
    if isinstance(payload, str):
        return _unique_tdx_symbols([payload])
    if isinstance(payload, Mapping):
        try:
            symbols = _symbols_from_tdx_kline_table(pd.DataFrame(payload))
            if symbols:
                return symbols
        except Exception:  # noqa: BLE001
            pass
        return _symbols_from_tdx_kline_mapping_values(payload)
    if isinstance(payload, (list, tuple, set)):
        values = list(payload)
        if not values:
            return []
        if all(isinstance(item, Mapping) for item in values):
            return _symbols_from_tdx_kline_table(pd.DataFrame(values))
        if all(isinstance(item, (str, int)) for item in values):
            return _unique_tdx_symbols(values)
        symbols: list[str] = []
        for item in values:
            symbols.extend(_symbols_from_tdx_kline_payload(item))
        return _unique_tdx_symbols(symbols)
    return []


def _symbols_from_tdx_kline_mapping_values(payload: Mapping[object, object]) -> list[str]:
    symbols: list[str] = []
    for value in payload.values():
        if isinstance(value, (pd.DataFrame, pd.Series, Mapping, list, tuple, set, str)):
            symbols.extend(_symbols_from_tdx_kline_payload(value))
    return _unique_tdx_symbols(symbols)


def _symbols_from_tdx_kline_table(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return []
    code_column = next((column for column in SYMBOL_COLUMNS if column in table.columns), None)
    if code_column is None:
        return []
    market_column = next((column for column in TDX_MARKET_COLUMNS if column in table.columns), None)
    symbols: list[str] = []
    for _, row in table.iterrows():
        market_hint = row[market_column] if market_column else None
        symbols.append(_normalize_tdx_list_symbol(row[code_column], market_hint=market_hint))
    return _unique_tdx_symbols(symbols)


def _normalize_tdx_list_symbol(value: object, *, market_hint: object | None = None) -> str:
    text = str(value).strip().upper().replace("_", ".")
    if not text:
        return ""
    if "." in text:
        return normalize_symbol(text)
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) < 6:
        return text
    code = digits[-6:]
    if code.startswith(TDX_BLOCK_INDEX_PREFIXES):
        return f"{code}.SH"
    exchange = _exchange_from_tdx_market(market_hint)
    if exchange:
        return f"{code}.{exchange}"
    if code.startswith("399"):
        return f"{code}.SZ"
    if code.startswith(("15", "16", "18")):
        return f"{code}.SZ"
    if code.startswith("5"):
        return f"{code}.SH"
    return normalize_symbol(code)


def _exchange_from_tdx_market(value: object | None) -> str:
    text = str(value).strip().upper()
    if not text or text in {"NONE", "NAN"}:
        return ""
    if text in {"1", "SH", "SSE"} or "上海" in text or text.startswith("SH"):
        return "SH"
    if text in {"0", "SZ", "SZSE"} or "深圳" in text or text.startswith("SZ"):
        return "SZ"
    if text in {"2", "BJ", "BSE"} or "北京" in text or "北交" in text or text.startswith("BJ"):
        return "BJ"
    return ""


def _unique_tdx_symbols(values: list[object]) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for value in values:
        symbol = _normalize_tdx_list_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def _symbols_from_mapping_values(payload: Mapping[object, object]) -> list[str]:
    symbols: list[str] = []
    for value in payload.values():
        if isinstance(value, (pd.DataFrame, pd.Series, Mapping, list, tuple, set, str)):
            symbols.extend(_symbols_from_tdx_stock_payload(value))
    return unique_symbols(symbols)


def _symbols_from_stock_table(table: pd.DataFrame) -> list[str]:
    try:
        return symbols_from_table(table)
    except ValueError:
        return []


def _filter_a_share_stock_symbols(symbols: list[str]) -> list[str]:
    filtered: list[str] = []
    for symbol in unique_symbols(symbols):
        if "." not in symbol:
            continue
        code, exchange = symbol.split(".", 1)
        if code.startswith(A_SHARE_STOCK_PREFIXES.get(exchange, ())):
            filtered.append(symbol)
    return filtered


def _filter_tdx_kline_symbols(symbols: list[str]) -> list[str]:
    filtered: list[str] = []
    for symbol in _unique_tdx_symbols(symbols):
        if _is_tdx_kline_symbol(symbol):
            filtered.append(symbol)
    return filtered


def _tdx_kline_symbol_table_from_payload(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table in _tdx_list_tables_from_payload(payload):
        if table.empty:
            continue
        code_column = next((column for column in SYMBOL_COLUMNS if column in table.columns), None)
        if code_column is None:
            continue
        market_column = next((column for column in TDX_MARKET_COLUMNS if column in table.columns), None)
        name_column = next((column for column in TDX_NAME_COLUMNS if column in table.columns), None)
        for _, row in table.iterrows():
            symbol = _normalize_tdx_list_symbol(row.get(code_column), market_hint=row.get(market_column) if market_column else None)
            if not _is_tdx_kline_symbol(symbol):
                continue
            name = str(row.get(name_column, "") if name_column else "").strip()
            rows.append({"symbol": symbol, "name": name, "category": _tdx_kline_category(symbol, name)})
    if not rows:
        return _empty_tdx_kline_symbol_table()
    return pd.DataFrame(rows, columns=TDX_KLINE_SYMBOL_TABLE_COLUMNS)


def _deduplicate_tdx_kline_symbol_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_tdx_kline_symbol_table()
    result = frame.copy()
    for column in TDX_KLINE_SYMBOL_TABLE_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result = result[TDX_KLINE_SYMBOL_TABLE_COLUMNS]
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["name"] = result["name"].fillna("").astype(str)
    result["category"] = result["category"].fillna("other").astype(str)
    return result.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def _empty_tdx_kline_symbol_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TDX_KLINE_SYMBOL_TABLE_COLUMNS)


def _is_tdx_kline_symbol(symbol: str) -> bool:
    if "." not in symbol:
        return False
    code, exchange = symbol.split(".", 1)
    return code.startswith(TDX_KLINE_PREFIXES.get(exchange, ()))


def _tdx_kline_category(symbol: str, name: str = "") -> str:
    if "." not in symbol:
        return "other"
    code, exchange = symbol.split(".", 1)
    if _is_tdx_etf(symbol, name) or (exchange in {"SH", "SZ"} and code.startswith(("5", "15", "16", "18"))):
        return "etf"
    if code.startswith(TDX_BLOCK_INDEX_PREFIXES) or (exchange == "SH" and code.startswith("000")) or (
        exchange == "SZ" and code.startswith("399")
    ):
        return "index"
    if exchange in A_SHARE_STOCK_PREFIXES and code.startswith(A_SHARE_STOCK_PREFIXES[exchange]):
        return "stock"
    return "other"


def _tdx_list_tables_from_payload(payload: Any) -> list[pd.DataFrame]:
    if payload is None:
        return []
    if isinstance(payload, pd.DataFrame):
        return [payload]
    if isinstance(payload, pd.Series):
        return [payload.to_frame(name="code")]
    if isinstance(payload, Mapping):
        try:
            table = pd.DataFrame(payload)
            if not table.empty:
                return [table]
        except Exception:  # noqa: BLE001
            pass
        tables: list[pd.DataFrame] = []
        for value in payload.values():
            tables.extend(_tdx_list_tables_from_payload(value))
        return tables
    if isinstance(payload, (list, tuple, set)):
        values = list(payload)
        if not values:
            return []
        if all(isinstance(item, Mapping) for item in values):
            return [pd.DataFrame(values)]
        tables: list[pd.DataFrame] = []
        for item in values:
            tables.extend(_tdx_list_tables_from_payload(item))
        return tables
    if isinstance(payload, (str, int)):
        return [pd.DataFrame({"code": [payload]})]
    return []


def _is_tdx_etf(symbol: str, name: str) -> bool:
    if not symbol or "." not in symbol:
        return False
    code, exchange = symbol.split(".", 1)
    if exchange not in {"SH", "SZ"}:
        return False
    if not code.startswith(("5", "15", "16", "18")):
        return False
    normalized_name = str(name or "").upper()
    return "ETF" in normalized_name or "交易型开放式" in normalized_name


def _normalize_etf_index(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "name", "amount", "category"]
    if frame.empty:
        return _empty_etf_index()
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = "" if column != "amount" else 0.0
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["name"] = result["name"].astype(str).str.strip()
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce").fillna(0.0)
    result["category"] = result["category"].astype(str).str.strip()
    result = result.loc[result["symbol"].ne("") & result["name"].ne("")]
    if result.empty:
        return _empty_etf_index()
    return result[columns].reset_index(drop=True)


def _empty_etf_index() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "name", "amount", "category"])


def _etf_category_key(name: object) -> str:
    text = _etf_query_key(name)
    for token in [
        "交易型开放式指数证券投资基金",
        "交易型开放式",
        "指数证券投资基金",
        "证券投资基金",
        "发起式联接",
        "联接",
        "增强",
        "基金",
        "ETF",
        "LOF",
    ]:
        text = text.replace(token.upper(), "")
    return text.strip("-_ ")


def _etf_query_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return re.sub(r"[\s　（）()【】\\[\\]：:·•,，、;；/\\\\-]+", "", text)


def _parse_tdx_amount(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(numeric) else float(numeric) * multiplier


def _batched_symbols(symbols: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size 至少需要 1。")
    return [symbols[index : index + batch_size] for index in range(0, len(symbols), batch_size)]


def _ensure_initialized(tq: Any) -> None:
    global _INITIALIZED, _INITIALIZED_CLIENT_ID
    client_id = id(tq)
    if _INITIALIZED and _INITIALIZED_CLIENT_ID == client_id:
        return
    try:
        tq.initialize(__file__)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TDX 初始化失败。请确认本机通达信终端已启动并登录；"
            f"目录只负责导入 tqcenter，终端仍需处于可连接状态。根因: {exc.__class__.__name__}: {exc}"
        ) from exc
    _INITIALIZED = True
    _INITIALIZED_CLIENT_ID = client_id


def _format_market_time(value: str) -> str:
    parsed = pd.Timestamp(pd.to_datetime(value))
    return parsed.strftime("%Y%m%d%H%M%S" if _has_explicit_time(value) else "%Y%m%d")


def _has_explicit_time(value: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}", str(value).strip()))


def _filter_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(pd.to_datetime(start))
    end_ts = pd.Timestamp(pd.to_datetime(end))
    if not _has_explicit_time(start):
        start_ts = start_ts.normalize()
    if not _has_explicit_time(end):
        end_ts = end_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return start_ts, end_ts


def _normalize_tdx_payload(
    raw_data: Any,
    *,
    symbol: str,
    start: str,
    end: str,
    allow_missing_symbol: bool = False,
) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(symbol)
    if raw_data is None:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"TDX 返回应为 dict[field]->DataFrame，实际为 {type(raw_data).__name__}。")
    payload = raw_data
    if not payload:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    selected_frames: dict[str, pd.DataFrame] = {}
    for field in REQUIRED_FIELDS:
        key = _resolve_field_key(payload, field)
        if key is None:
            raise ValueError(f"TDX 返回缺少必要字段：{field}")
        value = payload[key]
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"TDX 字段 {key} 应为 DataFrame，实际为 {type(value).__name__}。")
        columns = {str(column): column for column in value.columns}
        if normalized_symbol not in columns:
            if allow_missing_symbol:
                return pd.DataFrame(columns=CANONICAL_COLUMNS)
            raise ValueError(f"missing symbol column {normalized_symbol} in field {key}")
        selected_frames[field] = value

    if all(frame.empty for frame in selected_frames.values()):
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    series_map: dict[str, pd.Series] = {}
    for field, frame in selected_frames.items():
        column = {str(item): item for item in frame.columns}[normalized_symbol]
        series_map[field] = pd.Series(
            pd.to_numeric(frame[column].to_numpy(), errors="coerce"),
            index=pd.to_datetime(frame.index, errors="coerce"),
            name=field,
        )

    assembled = pd.concat(series_map, axis=1).reset_index().rename(columns={"index": "date"})
    assembled = assembled.rename(columns=OUTPUT_RENAME)
    assembled["date"] = pd.to_datetime(assembled["date"], errors="coerce")
    start_ts, end_ts = _filter_window(start, end)
    assembled = assembled.loc[assembled["date"].between(start_ts, end_ts)].copy()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        assembled[column] = pd.to_numeric(assembled[column], errors="coerce")
    assembled["stock_code"] = normalized_symbol
    assembled = assembled.dropna(subset=["date", "open", "high", "low", "close"])
    assembled = assembled[CANONICAL_COLUMNS].drop_duplicates(subset=["stock_code", "date"], keep="last")
    return assembled.sort_values("date").reset_index(drop=True)


def _resolve_field_key(payload: Mapping[str, Any], field: str) -> str | None:
    for key in FIELD_ALIASES[field]:
        if key in payload:
            return key
    return None
