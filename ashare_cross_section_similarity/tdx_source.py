from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.data import CANONICAL_COLUMNS
from ashare_cross_section_similarity.universe import normalize_symbol, symbols_from_table, unique_symbols

TDX_TQCENTER_ENV_VAR = "TDX_TQCENTER_PATH"
TDX_REQUEST_BATCH_SIZE = 100
TDX_SECTOR_INDEX_MARKET = "10"
TIMEFRAME_PERIODS = {"1d": "1d", "30m": "30m", "15m": "15m", "5m": "5m", "1m": "1m"}
ADJUST_MAP = {"": "none", "qfq": "front", "hfq": "back"}
REQUIRED_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount")
REFRESHABLE_KLINE_PERIODS = {"1d", "5m", "1m"}
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

_TQ_CLIENT: Any | None = None
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
        _refresh_tdx_kline_cache(tq, symbol_batch, period)
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
    _ensure_initialized(tq)

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
                return stock_symbols
            errors.append(f"{method_name}: 返回结果未包含 A 股股票代码")

    details = " | ".join(errors)
    raise RuntimeError(f"TDX 未能获取股票清单。请确认 tqcenter 支持股票列表接口。详情: {details}")


def fetch_tdx_sector_index_frame(*, tqcenter_path: str = "", tq_client: Any | None = None) -> pd.DataFrame:
    """Return TDX sector index codes using the documented get_sector_list contract."""
    tq = tq_client or _load_tq(tqcenter_path)
    _ensure_initialized(tq)

    errors: list[str] = []
    sector_list = getattr(tq, "get_sector_list", None)
    if callable(sector_list):
        frame = _tdx_code_name_frame_from_callable(
            "get_sector_list",
            sector_list,
            errors,
            call_variants=(((), {"list_type": 1}), ((), {})),
        )
        if not frame.empty:
            return frame

    stock_list = getattr(tq, "get_stock_list", None)
    if callable(stock_list):
        frame = _tdx_code_name_frame_from_callable(
            "get_stock_list",
            stock_list,
            errors,
            call_variants=(((TDX_SECTOR_INDEX_MARKET,), {"list_type": 1}), ((TDX_SECTOR_INDEX_MARKET,), {})),
        )
        if not frame.empty:
            return frame

    details = " | ".join(errors)
    raise RuntimeError(
        "TDX 未能获取板块指数列表。请确认 tqcenter 支持 get_sector_list 或 get_stock_list('10')。"
        f" 详情: {details}"
    )


def fetch_tdx_sector_index_symbols(*, tqcenter_path: str = "", tq_client: Any | None = None) -> list[str]:
    frame = fetch_tdx_sector_index_frame(tqcenter_path=tqcenter_path, tq_client=tq_client)
    return frame["symbol"].dropna().astype(str).tolist()


def _load_tq(tqcenter_path: str = "") -> Any:
    global _TQ_CLIENT
    if _TQ_CLIENT is not None:
        return _TQ_CLIENT

    errors: list[str] = []
    for path in _candidate_import_paths(tqcenter_path):
        resolved = path.resolve()
        if not resolved.exists():
            errors.append(f"{resolved} 不存在")
            continue
        inserted = False
        path_text = str(resolved)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted = True
        try:
            module = importlib.import_module("tqcenter")
            _TQ_CLIENT = getattr(module, "tq")
            return _TQ_CLIENT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_text}: {exc}")
            if inserted:
                try:
                    sys.path.remove(path_text)
                except ValueError:
                    pass

    try:
        module = importlib.import_module("tqcenter")
        _TQ_CLIENT = getattr(module, "tq")
        return _TQ_CLIENT
    except Exception as exc:  # noqa: BLE001
        errors.append(f"normal import: {exc}")
        details = " | ".join(errors)
        raise RuntimeError(
            "无法导入 tqcenter。请先安装并登录本机通达信终端，并通过"
            f" {TDX_TQCENTER_ENV_VAR} 或下载源输入框指向 TDX 的 PYPlugins/user 目录。"
            f" 详情: {details}"
        ) from exc


def _candidate_import_paths(tqcenter_path: str = "") -> list[Path]:
    raw_value = (tqcenter_path or os.getenv(TDX_TQCENTER_ENV_VAR, "")).strip()
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
    for item in raw_value.split(os.pathsep):
        text = item.strip().strip('"')
        if not text:
            continue
        for path in expand(Path(text).expanduser()):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


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


def _tdx_code_name_frame_from_callable(
    method_name: str,
    method: Any,
    errors: list[str],
    *,
    call_variants: tuple[tuple[tuple[object, ...], dict[str, object]], ...],
) -> pd.DataFrame:
    for args, kwargs in call_variants:
        label = _call_label(args, kwargs)
        try:
            payload = method(*args, **kwargs)
        except TypeError as exc:
            errors.append(f"{method_name}{label}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{method_name}{label}: {exc}")
            continue
        frame = _tdx_code_name_frame(payload)
        if not frame.empty:
            return frame
        errors.append(f"{method_name}{label}: 返回结果未包含板块指数代码")
    return pd.DataFrame(columns=["symbol", "name"])


def _call_label(args: tuple[object, ...], kwargs: dict[str, object]) -> str:
    parts = [repr(item) for item in args]
    parts.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return f"({', '.join(parts)})"


def _tdx_code_name_frame(payload: Any) -> pd.DataFrame:
    records = _tdx_code_name_records(payload)
    if not records:
        return pd.DataFrame(columns=["symbol", "name"])
    frame = pd.DataFrame(records)
    frame["symbol"] = frame["symbol"].map(normalize_tdx_sector_index_symbol)
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["symbol"].astype(str).str.match(r"^\d{6}\.[A-Z]{2}$", na=False)]
    return frame.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def _tdx_code_name_records(payload: Any) -> list[dict[str, str]]:
    if payload is None:
        return []
    if isinstance(payload, pd.DataFrame):
        return _tdx_code_name_records(payload.to_dict("records"))
    if isinstance(payload, pd.Series):
        return _tdx_code_name_records(payload.dropna().tolist())
    if isinstance(payload, str):
        return [{"symbol": payload, "name": ""}]
    if isinstance(payload, Mapping):
        code = _mapping_first_value(payload, ("Code", "code", "stock_code", "symbol", "证券代码", "代码"))
        name = _mapping_first_value(payload, ("Name", "name", "stock_name", "证券名称", "名称"))
        if code:
            return [{"symbol": str(code), "name": str(name or "")}]
        records: list[dict[str, str]] = []
        for value in payload.values():
            if isinstance(value, (pd.DataFrame, pd.Series, Mapping, list, tuple, set, str)):
                records.extend(_tdx_code_name_records(value))
        return records
    if isinstance(payload, (list, tuple, set)):
        records: list[dict[str, str]] = []
        for item in payload:
            records.extend(_tdx_code_name_records(item))
        return records
    return []


def _mapping_first_value(payload: Mapping[object, object], keys: tuple[str, ...]) -> object:
    exact = {str(key): value for key, value in payload.items()}
    lower = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key in exact:
            return exact[key]
        if key.lower() in lower:
            return lower[key.lower()]
    return ""


def normalize_tdx_sector_index_symbol(value: object) -> str:
    text = str(value).strip().upper().replace("_", ".")
    if not text:
        return ""
    if "." in text:
        code, exchange = text.split(".", 1)
        digits = "".join(character for character in code if character.isdigit())
        code = digits[-6:].zfill(6) if digits else code
        return f"{code}.{exchange[:2]}"
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 6 and digits[-6:].startswith("88"):
        return f"{digits[-6:]}.SH"
    return normalize_symbol(text)


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
        raise RuntimeError("TDX 初始化失败。请确认本机通达信终端已启动并登录。") from exc
    _INITIALIZED = True
    _INITIALIZED_CLIENT_ID = client_id


def _refresh_tdx_kline_cache(tq: Any, symbols: list[str], period: str) -> None:
    if period not in REFRESHABLE_KLINE_PERIODS:
        return
    refresh = getattr(tq, "refresh_kline", None)
    if not callable(refresh):
        return
    try:
        result = refresh(list(symbols), period)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"TDX K线缓存刷新失败：{exc}") from exc
    error = _tdx_refresh_error(result)
    if error:
        raise RuntimeError(f"TDX K线缓存刷新失败：{error}")


def _tdx_refresh_error(result: object) -> str:
    if result is None:
        return "接口无返回"
    if isinstance(result, Mapping):
        payload = result
    elif isinstance(result, str):
        text = result.strip()
        if not text:
            return "接口返回为空"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
    else:
        return ""
    error_id = str(payload.get("ErrorId", "0"))
    if error_id in {"", "0", "None"}:
        return ""
    return str(payload.get("Error") or payload.get("Msg") or payload)


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
