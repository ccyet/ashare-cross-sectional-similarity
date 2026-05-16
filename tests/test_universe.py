from __future__ import annotations

import pandas as pd

from ashare_cross_section_similarity.universe import normalize_symbol, symbols_from_table


def test_normalize_symbol_adds_exchange_suffix_for_a_share_codes() -> None:
    assert normalize_symbol("000001") == "000001.SZ"
    assert normalize_symbol("399006") == "399006.SZ"
    assert normalize_symbol("600519") == "600519.SH"
    assert normalize_symbol("000300.SH") == "000300.SH"


def test_symbols_from_table_accepts_common_chinese_column_names() -> None:
    table = pd.DataFrame({"证券代码": ["000001", "600519"], "名称": ["平安银行", "贵州茅台"]})

    assert symbols_from_table(table) == ["000001.SZ", "600519.SH"]
