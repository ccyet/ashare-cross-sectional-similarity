from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.universe import (
    DEFAULT_ANALYSIS_INDEX_SYMBOLS,
    normalize_symbol,
    symbols_from_table,
    symbols_with_analysis_indexes,
)


def test_normalize_symbol_adds_exchange_suffix_for_a_share_codes() -> None:
    assert normalize_symbol("000001") == "000001.SZ"
    assert normalize_symbol("399006") == "399006.SZ"
    assert normalize_symbol("600519") == "600519.SH"
    assert normalize_symbol("920006") == "920006.BJ"
    assert normalize_symbol("000300.SH") == "000300.SH"


def test_symbols_with_analysis_indexes_includes_common_index_proxies_by_default() -> None:
    symbols = symbols_with_analysis_indexes(["000001.SZ", "600519.SH"])

    assert symbols[:2] == ["000001.SZ", "600519.SH"]
    assert "399006.SZ" in symbols
    assert "000300.SH" in symbols
    assert "000852.SH" in symbols
    assert all(symbol in symbols for symbol in DEFAULT_ANALYSIS_INDEX_SYMBOLS)


def test_symbols_with_analysis_indexes_can_add_extra_symbols_and_deduplicate() -> None:
    symbols = symbols_with_analysis_indexes(
        ["000001.SZ"],
        include_indexes=False,
        extra_symbols=["399006", "000300.SH", "399006.SZ"],
    )

    assert symbols == ["000001.SZ", "399006.SZ", "000300.SH"]


def test_symbols_from_table_accepts_common_chinese_column_names() -> None:
    table = pd.DataFrame({"证券代码": ["000001", "600519"], "名称": ["平安银行", "贵州茅台"]})

    assert symbols_from_table(table) == ["000001.SZ", "600519.SH"]
