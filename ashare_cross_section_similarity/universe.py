from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SYMBOL_COLUMNS = (
    "code",
    "stock_code",
    "symbol",
    "ts_code",
    "证券代码",
    "代码",
    "成分券代码",
    "品种代码",
    "股票代码",
)


def normalize_symbol(value: object) -> str:
    text = str(value).strip().upper()
    if not text:
        return ""
    text = text.replace("_", ".")
    if "." in text:
        code, exchange = text.split(".", 1)
        return f"{code.zfill(6)}.{exchange[:2]}"
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) < 6:
        return text
    code = digits[-6:]
    if code.startswith("920"):
        exchange = "BJ"
    elif code.startswith(("6", "5", "9")):
        exchange = "SH"
    elif code.startswith(("4", "8")):
        exchange = "BJ"
    else:
        exchange = "SZ"
    return f"{code}.{exchange}"


def unique_symbols(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for value in values:
        symbol = normalize_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def symbols_from_table(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return []
    for column in SYMBOL_COLUMNS:
        if column in table.columns:
            return unique_symbols(table[column].dropna().tolist())
    raise ValueError(f"未找到证券代码列，支持列名：{', '.join(SYMBOL_COLUMNS)}")


def load_universe_file(path: str | Path) -> list[str]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"universe 文件不存在：{file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        table = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        table = pd.read_excel(file_path)
    elif suffix == ".parquet":
        table = pd.read_parquet(file_path)
    else:
        raise ValueError("universe 文件仅支持 csv、xlsx、xls、parquet。")
    return symbols_from_table(table)


def fetch_index_constituents(index_code: str) -> list[str]:
    import akshare as ak

    code = normalize_symbol(index_code).split(".", 1)[0]
    errors: list[str] = []
    for func_name in ("index_stock_cons_csindex", "index_stock_cons"):
        try:
            table = getattr(ak, func_name)(symbol=code)
            symbols = symbols_from_table(table)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{func_name}: {exc}")
            continue
        if symbols:
            return symbols
    raise RuntimeError("指数成分获取失败：" + " | ".join(errors))


def fetch_industry_constituents(name_or_code: str) -> list[str]:
    import akshare as ak

    table = ak.stock_board_industry_cons_em(symbol=name_or_code)
    return symbols_from_table(table)


def fetch_concept_constituents(name_or_code: str) -> list[str]:
    import akshare as ak

    table = ak.stock_board_concept_cons_em(symbol=name_or_code)
    return symbols_from_table(table)
