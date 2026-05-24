from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget, QTableView, QTextEdit

from ashare_cross_section_similarity.desktop.app import (
    DatePicker,
    DataFrameModel,
    MainWindow,
    _multi_table_tab,
    _review_card_palette,
    _review_card_widget,
    _style_sheet,
    _table,
    parse_kline_grid,
)
from ashare_cross_section_similarity.desktop.view_model import ReviewCritiqueCard
from ashare_cross_section_similarity.review import ReviewConfig, analyze_price_review


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_date_picker_uses_readonly_text_and_popup_calendar() -> None:
    _app()

    picker = DatePicker()

    assert picker.editor.isReadOnly()
    assert picker.calendar is not None
    assert picker.date_text().count("-") == 2


def test_parse_kline_grid_accepts_product_labels() -> None:
    assert parse_kline_grid("1*2") == (1, 2)
    assert parse_kline_grid("2*2") == (2, 2)
    assert parse_kline_grid("3*3") == (3, 3)


def test_main_window_separates_charts_and_tables_for_readability() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()

    assert isinstance(window.history_result_tabs, QTabWidget)
    assert window.history_result_tabs.tabText(0) == "图表"
    assert window.history_result_tabs.tabText(1) == "统计"
    assert window.history_result_tabs.tabText(2) == "数据"
    assert window.history_result_tabs.tabText(3) == "表格"
    assert window.history_size_spread_chart.title == "大小盘价差率"
    assert window.history_size_spread_table.minimumHeight() >= 180
    assert window.history_download_start.text() == "2018-01-01"
    assert window.history_download_button.text() == "下载/更新该标的行情"
    assert window.history_latest_button.text() == "最新收盘"
    assert window.cross_result_tabs.tabText(1) == "统计"
    assert window.cross_result_tabs.tabText(2) == "数据"
    assert window.cross_check_button.text() == "检查本地数据覆盖"
    assert window.cross_download_button.text() == "检查并下载缺失行情"
    assert window.cross_data_chart.title == "本地覆盖状态"
    assert window.review_result_tabs.tabText(0) == "图表/锐评"
    assert window.review_result_tabs.tabText(1) == "统计"
    assert window.review_load_etf_button.text() == "加载主要ETF"
    assert window.review_add_etf_button.text() == "加入选中ETF"
    assert window.review_popular_etf_table.minimumHeight() >= 180
    assert window.review_script_table.minimumHeight() >= 180
    assert window.history_table.minimumHeight() >= 360
    assert window.history_table_expand_button.text() == "展开表格"
    assert window.history_kline_layout.currentText() == "2*2"
    assert window.history_kline_layout.isEditable()
    assert window.history_kline_expand_button.text() == "展开K线"
    assert window.data_update_button.text() == "下载/更新行情"
    assert window.full_daily_plan_button.text() == "预览全A日线"
    assert window.full_daily_run_button.text() == "开始全A下载"
    assert window.price_import_button.text() == "导入价格数据"
    assert window.price_import_table.minimumHeight() >= 180
    assert "python -m ashare_cross_section_similarity benchmark" in window.benchmark_command.toPlainText()
    assert [window.data_result_tabs.tabText(index) for index in range(window.data_result_tabs.count())] == [
        "覆盖",
        "价格导入",
        "全A日线",
        "K线迁移",
    ]
    assert window.review_auto_etf_table.minimumHeight() >= 180


def test_stat_sections_use_inner_tabs_instead_of_stacked_tables() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()

    history_stats_tabs = window.history_result_tabs.widget(1).findChild(QTabWidget)
    review_stats_tabs = window.review_result_tabs.widget(1).findChild(QTabWidget)

    assert history_stats_tabs is not None
    assert [history_stats_tabs.tabText(index) for index in range(history_stats_tabs.count())] == [
        "后验观察统计",
        "相似度分层表现",
        "大小盘价差率",
    ]
    assert review_stats_tabs is not None
    assert review_stats_tabs.tabText(3) == "视频脚本视角"
    assert review_stats_tabs.tabText(4) == "ETF自动代理"


def test_full_daily_progress_updates_status_text() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()

    window._show_full_daily_progress(
        {"completed": 25, "total": 100, "batch_index": 2, "batch_count": 8, "current": "000001.SZ, 600519.SH"}
    )

    assert "全A下载进度：25/100" in window.full_daily_status.text()
    assert "批次 2/8" in window.full_daily_status.text()


def test_main_window_exposes_original_search_parameters() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()

    assert window.history_forward_windows.text() == "5,20,60"
    assert window.history_candidate_n.value() == 100
    assert window.history_exclusion.value() == 20
    assert window.history_gap.value() == 20
    assert window.cross_min_coverage.value() == 0.8
    assert window.cross_path_weight.value() == 0.7
    assert window.review_min_swing.value() == 5.0
    assert window.review_min_segment.value() == 3


def test_combo_boxes_use_flat_dropdown_style() -> None:
    style = _style_sheet()

    assert "QComboBox {\n    background: transparent;" in style
    assert "border-radius: 0;" in style
    assert "QComboBox::drop-down {\n    border: none;\n    width: 0px;" in style
    assert "QComboBox::down-arrow {\n    image: none;" in style


def test_kline_layout_control_has_spacing_shortcuts_and_expand_toggle() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()

    combo = window.history_kline_layout
    line_edit = combo.lineEdit()

    assert combo.minimumWidth() >= 96
    assert line_edit is not None
    assert line_edit.textMargins().left() >= 8
    assert combo.itemText(2) == "2*4"
    assert [button.text() for button in window.history_kline_toolbar.findChildren(QPushButton, "klineLayoutShortcut")] == [
        "1*2",
        "2*2",
        "2*4",
        "3*3",
    ]

    before_height = window.history_kline_chart.minimumHeight()
    shortcut = window.history_kline_toolbar.findChildren(QPushButton, "klineLayoutShortcut")[2]
    shortcut.click()

    assert combo.currentText() == "2*4"
    assert window.history_kline_chart._grid_rows == 2
    assert window.history_kline_chart._grid_columns == 4

    window.history_kline_expand_button.click()

    assert window.history_kline_expand_button.text() == "收起K线"
    assert window.history_kline_chart.minimumHeight() > before_height


def test_review_card_uses_source_grade_palette() -> None:
    card = ReviewCritiqueCard(
        symbol="300750.SZ",
        title="宁德时代（300750.SZ）",
        grade="夯爆了",
        nature="趋势启动",
        critique="弹性足。",
        metrics=(("收益", "12.00%"), ("回撤", "-3.00%")),
    )

    palette = _review_card_palette(card.grade)
    widget = _review_card_widget(card)

    assert palette.accent == "#7c3aed"
    assert palette.background == "#f5f3ff"
    assert "#7c3aed" in widget.styleSheet()
    assert "#f5f3ff" in widget.styleSheet()
    assert "#5b21b6" in "".join(child.styleSheet() for child in widget.findChildren(QLabel))


def test_review_text_output_can_fold_and_scroll_horizontally() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()

    assert window.review_text.lineWrapMode() == QTextEdit.LineWrapMode.NoWrap
    assert window.review_text.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert window.review_text_toggle_button.text() == "收起文本"

    window.review_text_toggle_button.click()

    assert window.review_text_toggle_button.text() == "展开文本"
    assert not window.review_text.isVisible()


def test_review_popular_etfs_open_visible_candidate_dialog() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()
    frame = pd.DataFrame(
        [
            {"symbol": "510300.SH", "name": "沪深300ETF"},
            {"symbol": "159915.SZ", "name": "创业板ETF"},
        ]
    )

    window._show_review_etf_candidates(frame)
    app.processEvents()

    assert isinstance(window.review_popular_etf_table.model(), DataFrameModel)
    assert window.review_etf_dialog is not None
    assert window.review_etf_dialog.windowTitle() == "主要ETF候选"
    popup_table = window.review_etf_dialog.findChild(QTableView, "reviewEtfPopupTable")
    assert popup_table is not None
    assert isinstance(popup_table.model(), DataFrameModel)
    assert popup_table.model().rowCount() == 2


def test_windows_fill_fields_use_explicit_dark_text_color() -> None:
    style = _style_sheet()

    assert "QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit" in style
    assert "color: #17241e;" in style
    assert "QComboBox QLineEdit" in style
    assert "QTableView#resultTable" in style and "selection-color: #ffffff;" in style


def test_table_cells_are_centered_and_not_elided() -> None:
    model = DataFrameModel(pd.DataFrame({"代码": ["300750.SZ"], "锐评结论": ["长文本需要完整查看"]}))
    alignment = model.data(model.index(0, 0), Qt.ItemDataRole.TextAlignmentRole)
    table = _table()

    assert alignment & Qt.AlignmentFlag.AlignHCenter
    assert alignment & Qt.AlignmentFlag.AlignVCenter
    assert table.textElideMode() == Qt.TextElideMode.ElideNone
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn


def test_stat_table_tabs_offer_expandable_full_width_dialog_action() -> None:
    table = _table()
    table.setModel(DataFrameModel(pd.DataFrame({"长字段": ["这是一段需要横向滚动查看的完整内容"]})))

    tab = _multi_table_tab((("排序总表", table),))
    buttons = tab.findChildren(QPushButton)

    assert any(button.text() == "展开统计" for button in buttons)


def test_review_cards_are_one_per_symbol_and_use_name_direction_mapping() -> None:
    app = _app()
    window = MainWindow()
    app.processEvents()
    reviews = [
        analyze_price_review(
            _bars("300750.SZ", [10, 11, 12, 13, 14, 15]),
            ReviewConfig(symbol="300750.SZ", start="2024-01-01", end="2024-01-06"),
        ),
        analyze_price_review(
            _bars("600519.SH", [20, 19, 18, 17, 16, 15]),
            ReviewConfig(symbol="600519.SH", start="2024-01-01", end="2024-01-06"),
        ),
        analyze_price_review(
            _bars("000001.SZ", [9, 9.5, 10, 10.5, 11, 11.5]),
            ReviewConfig(symbol="000001.SZ", start="2024-01-01", end="2024-01-06"),
        ),
    ]
    script_profiles = [
        {"代码": "300750.SZ", "股票": "宁德时代", "强弱等级": "人上人", "当前性质": "趋势启动", "锐评结论": "强。"},
        {"代码": "600519.SH", "股票": "贵州茅台", "强弱等级": "NPC", "当前性质": "走弱", "锐评结论": "弱。"},
        {"代码": "000001.SZ", "股票": "平安银行", "强弱等级": "立棍单打", "当前性质": "趋势启动", "锐评结论": "稳。"},
    ]
    bundle = SimpleNamespace(
        reviews=reviews,
        comparison_frame=pd.DataFrame(),
        comparison_frames=[],
        script_profiles=script_profiles,
        etf_matches=pd.DataFrame(),
        stock_names={"300750.SZ": "宁德时代", "600519.SH": "贵州茅台", "000001.SZ": "平安银行"},
        direction_by_symbol={symbol: "行业:新能源 / 概念:锂电" for symbol in ("300750.SZ", "600519.SH", "000001.SZ")},
    )

    window._show_review_result({"bundle": bundle})
    app.processEvents()

    cards = window.review_cards.findChildren(QLabel, "reviewCardTitle")
    ranking = window.review_ranking_table.model().frame()

    assert len(cards) == 3
    assert {card.text() for card in cards} == {"宁德时代（300750.SZ）", "贵州茅台（600519.SH）", "平安银行（000001.SZ）"}
    assert set(ranking["股票"]) == {"宁德时代", "贵州茅台", "平安银行"}
    assert set(ranking["所属方向"]) == {"行业:新能源 / 概念:锂电"}


def _bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "stock_code": symbol,
            "open": closes,
            "high": [value * 1.02 for value in closes],
            "low": [value * 0.98 for value in closes],
            "close": closes,
            "volume": list(range(100, 100 + len(closes))),
            "amount": list(range(1000, 1000 + len(closes))),
        }
    )
