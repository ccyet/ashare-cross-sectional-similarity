from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import re
import threading
from typing import Any

import pandas as pd

from ashare_cross_section_similarity.desktop.ai import run_review_ai
from ashare_cross_section_similarity.desktop.app_icon import create_app_icon
from ashare_cross_section_similarity.desktop.service import (
    CrossSectionRequest,
    DataCoverageRequest,
    DataUpdateRequest,
    DesktopAppConfig,
    DesktopSearchService,
    FullDailyUpdateRequest,
    HistoryRequest,
    MissingDataUpdateRequest,
    PriceImportRequest,
    ReviewRequest,
    export_frame_csv,
    default_desktop_data_root,
    import_symbol_file,
    parse_symbol_list,
)
from ashare_cross_section_similarity.desktop.view_model import (
    BarValue,
    CandlestickSeries,
    LineSeries,
    OverviewModule,
    ReviewCritiqueCard,
    cross_section_candlestick_series,
    cross_section_score_bars,
    cross_section_stat_frames,
    data_status_bars,
    data_status_frame,
    history_candlestick_series,
    history_line_series,
    history_stat_frames,
    overview_modules,
    review_script_profile_frame,
    review_candlestick_series,
    review_comparisons_frame,
    review_critique_cards,
    review_overview_frame,
    review_ranking_frame,
    review_relative_line_series,
    review_segments_frame,
    review_text,
    size_spread_frame,
    size_spread_line_series,
    size_spread_window_stats,
    visible_data_fields,
)
from ashare_cross_section_similarity.data import resolve_timeframe_root
from ashare_cross_section_similarity.llm_client import DEFAULT_LLM_PROVIDER, LLMConfig, provider_presets
from ashare_cross_section_similarity.similarity_algorithms import ALGORITHM_CHOICES, BASELINE_ALGORITHM, algorithm_label

try:
    from PySide6.QtCore import QAbstractTableModel, QDate, QModelIndex, QPoint, QRectF, QSortFilterProxyModel, Qt, QThread, Signal
    from PySide6.QtGui import QColor, QCloseEvent, QMouseEvent, QPainter, QPen, QWheelEvent
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCalendarWidget,
        QComboBox,
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QStackedWidget,
        QTabWidget,
        QTableView,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised only when desktop deps are absent
    raise RuntimeError("Qt 桌面应用需要安装 PySide6：pip install PySide6") from exc


class DataFrameModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        super().__init__()
        self._frame = pd.DataFrame() if frame is None else frame.reset_index(drop=True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or role not in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.TextAlignmentRole}:
            return None
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        column = str(self._frame.columns[index.column()])
        return _format_cell(self._frame.iat[index.row(), index.column()], column)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._frame.columns[section])
        return str(section + 1)

    def frame(self) -> pd.DataFrame:
        return self._frame.copy()


class DataFrameFilterProxyModel(QSortFilterProxyModel):
    def __init__(self) -> None:
        super().__init__()
        self._search_text = ""
        self._status_filter = "全部"

    def set_search_text(self, value: str) -> None:
        self._search_text = str(value or "").strip().lower()
        self.invalidateFilter()

    def set_status_filter(self, value: str) -> None:
        self._status_filter = str(value or "全部").strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, DataFrameModel):
            return super().filterAcceptsRow(source_row, source_parent)
        frame = model.frame()
        if source_row >= len(frame):
            return False
        row = frame.iloc[source_row]
        if self._status_filter and self._status_filter != "全部":
            status = str(row.get("status", row.get("状态", "")) or "")
            if status != self._status_filter:
                return False
        if not self._search_text:
            return True
        return self._search_text in " ".join(str(value) for value in row.tolist()).lower()


class TaskWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    progressed = Signal(object)

    def __init__(self, operation: Callable[..., object], *, progress_enabled: bool = False) -> None:
        super().__init__()
        self._operation = operation
        self._progress_enabled = progress_enabled

    def run(self) -> None:
        try:
            if self._progress_enabled:
                self.succeeded.emit(self._operation(self.progressed.emit))
            else:
                self.succeeded.emit(self._operation())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class DatePicker(QWidget):
    def __init__(self, value: QDate | None = None) -> None:
        super().__init__()
        self.editor = QLineEdit()
        self.editor.setReadOnly(True)
        self.editor.setCursor(Qt.CursorShape.PointingHandCursor)
        self.editor.mousePressEvent = self._open_from_editor  # type: ignore[method-assign]
        self.button = QPushButton("选择")
        self.button.setObjectName("datePickButton")
        self.button.clicked.connect(self.open_calendar)
        self.calendar = QCalendarWidget()
        _configure_calendar(self.calendar)
        self._date = value or QDate.currentDate()
        self._sync_text()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.editor, stretch=1)
        layout.addWidget(self.button)

    def date(self) -> QDate:
        return QDate(self._date)

    def setDate(self, value: QDate) -> None:  # noqa: N802
        if not value.isValid():
            return
        self._date = QDate(value)
        self._sync_text()

    def date_text(self) -> str:
        return self._date.toString("yyyy-MM-dd")

    def open_calendar(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("选择日期")
        dialog.setModal(True)
        calendar = QCalendarWidget(dialog)
        _configure_calendar(calendar)
        calendar.setSelectedDate(self._date)
        self.calendar = calendar
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(calendar)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.setDate(calendar.selectedDate())

    def _open_from_editor(self, event: object) -> None:  # noqa: ARG002
        self.open_calendar()

    def _sync_text(self) -> None:
        self.editor.setText(self.date_text())


class LineChartWidget(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.series: list[LineSeries] = []
        self.setMinimumHeight(210)
        self.setObjectName("chartWidget")

    def set_series(self, series: list[LineSeries]) -> None:
        self.series = series
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_panel(painter, self.rect())
        _draw_chart_title(painter, self.title, self.rect())
        plot = self.rect().adjusted(18, 42, -18, -24)
        if not self.series:
            _draw_empty_chart(painter, plot, "暂无图表数据")
            return
        values = [value for item in self.series for value in item.values]
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            min_value -= 1
            max_value += 1
        _draw_grid(painter, plot)
        palette = _chart_palette()
        for index, item in enumerate(self.series):
            color = palette[index % len(palette)]
            pen = QPen(color, 2)
            painter.setPen(pen)
            points = _line_points(item.values, plot, min_value, max_value)
            for start, end in zip(points, points[1:]):
                painter.drawLine(start[0], start[1], end[0], end[1])
        _draw_legend(painter, self.series, palette, self.rect())


class BarChartWidget(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.bars: list[BarValue] = []
        self.setMinimumHeight(210)
        self.setObjectName("chartWidget")

    def set_bars(self, bars: list[BarValue]) -> None:
        self.bars = bars
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_panel(painter, self.rect())
        _draw_chart_title(painter, self.title, self.rect())
        plot = self.rect().adjusted(18, 42, -18, -18)
        if not self.bars:
            _draw_empty_chart(painter, plot, "暂无图表数据")
            return
        max_value = max(max(item.value for item in self.bars), 1.0)
        row_height = max(18, min(28, plot.height() // max(len(self.bars), 1)))
        for index, item in enumerate(self.bars):
            y = plot.top() + index * row_height
            label_width = 82
            bar_left = plot.left() + label_width
            bar_width = int((plot.width() - label_width - 54) * max(item.value, 0) / max_value)
            painter.setPen(QColor("#52605a"))
            painter.drawText(plot.left(), y + row_height - 7, item.label)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#dfe8e1"))
            painter.drawRoundedRect(bar_left, y + 5, plot.width() - label_width - 54, 10, 5, 5)
            painter.setBrush(QColor("#2d6a4f"))
            painter.drawRoundedRect(bar_left, y + 5, bar_width, 10, 5, 5)
            painter.setPen(QColor("#52605a"))
            painter.drawText(plot.right() - 44, y + row_height - 7, f"{item.value:.2f}")


class ReviewCardsWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout = QGridLayout(self)
        self._dialogs: list[QDialog] = []
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(12)
        self._layout.setVerticalSpacing(12)
        self.setObjectName("reviewCards")
        self.set_cards([])

    def set_cards(
        self,
        cards: list[ReviewCritiqueCard],
        *,
        chart_series: list[CandlestickSeries] | None = None,
    ) -> None:
        _clear_layout(self._layout)
        if not cards:
            empty = QLabel("暂无锐评卡片")
            empty.setObjectName("mutedText")
            self._layout.addWidget(empty, 0, 0)
            return
        series_by_symbol = _series_by_symbol(chart_series or [])
        for index, card in enumerate(cards[:9]):
            self._layout.addWidget(
                _review_card_widget(
                    card,
                    on_expand=lambda item=card, series=series_by_symbol.get(card.symbol, []): self._open_card_dialog(item, series),
                ),
                index // 3,
                index % 3,
            )
        for column in range(3):
            self._layout.setColumnStretch(column, 1)

    def _open_card_dialog(self, card: ReviewCritiqueCard, chart_series: list[CandlestickSeries]) -> None:
        dialog = _review_kline_dialog(card, chart_series, self)
        self._dialogs.append(dialog)
        dialog.finished.connect(lambda _=0, item=dialog: self._forget_dialog(item))
        dialog.showMaximized()

    def _forget_dialog(self, dialog: QDialog) -> None:
        if dialog in self._dialogs:
            self._dialogs.remove(dialog)


class CandlestickChartWidget(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.series: list[CandlestickSeries] = []
        self._visible_left = 0.0
        self._visible_right = 1.0
        self._grid_rows = 2
        self._grid_columns = 2
        self._expanded = False
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setObjectName("chartWidget")

    def set_grid(self, rows: int, columns: int) -> None:
        self._grid_rows = max(1, int(rows))
        self._grid_columns = max(1, int(columns))
        self._sync_minimum_height()
        self.update()

    def set_series(self, series: list[CandlestickSeries]) -> None:
        self.series = series
        self._visible_left = 0.0
        self._visible_right = 1.0
        self._sync_minimum_height()
        self.update()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self._sync_minimum_height()
        self.updateGeometry()
        self.update()

    def is_expanded(self) -> bool:
        return self._expanded

    def _sync_minimum_height(self) -> None:
        row_height = 320 if self._expanded else 230
        minimum = 420 if self._expanded else 260
        maximum = 1400 if self._expanded else 900
        self.setMinimumHeight(max(minimum, min(maximum, row_height * self._grid_rows)))

    def paintEvent(self, event: object) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        _draw_panel(painter, self.rect())
        _draw_chart_title(painter, self.title, self.rect())
        chart_area = self.rect().adjusted(16, 42, -16, -18)
        if not self.series:
            _draw_empty_chart(painter, chart_area, "暂无K线数据")
            return
        gap = 12
        visible_series = self.series[: self._grid_rows * self._grid_columns]
        panel_width = max(180, (chart_area.width() - gap * (self._grid_columns - 1)) / max(1, self._grid_columns))
        panel_height = max(150, (chart_area.height() - gap * (self._grid_rows - 1)) / max(1, self._grid_rows))
        for index, item in enumerate(visible_series):
            row = index // self._grid_columns
            column = index % self._grid_columns
            left = chart_area.left() + column * (panel_width + gap)
            top = chart_area.top() + row * (panel_height + gap)
            panel = QRectF(left, top, panel_width, panel_height)
            _draw_candlestick_panel(painter, panel, item, self._visible_left, self._visible_right)
        if self._drag_start and self._drag_current:
            left = min(self._drag_start.x(), self._drag_current.x())
            right = max(self._drag_start.x(), self._drag_current.x())
            painter.setPen(QPen(QColor("#2d6a4f"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(45, 106, 79, 34))
            painter.drawRect(left, chart_area.top(), right - left, chart_area.height())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        span = self._visible_right - self._visible_left
        if span <= 0:
            return
        factor = 0.82 if event.angleDelta().y() > 0 else 1.18
        new_span = min(1.0, max(0.08, span * factor))
        center = (self._visible_left + self._visible_right) / 2
        self._set_visible_range(center - new_span / 2, center + new_span / 2)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._drag_current = self._drag_start
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_start is not None:
            self._drag_current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return
        end = event.position().toPoint()
        width = abs(end.x() - self._drag_start.x())
        if width >= 12:
            chart_area = self.rect().adjusted(16, 42, -16, -18)
            left = min(self._drag_start.x(), end.x())
            right = max(self._drag_start.x(), end.x())
            next_left = _clamp((left - chart_area.left()) / max(chart_area.width(), 1), 0.0, 1.0)
            next_right = _clamp((right - chart_area.left()) / max(chart_area.width(), 1), 0.0, 1.0)
            visible_span = self._visible_right - self._visible_left
            self._set_visible_range(
                self._visible_left + visible_span * next_left,
                self._visible_left + visible_span * next_right,
            )
        self._drag_start = None
        self._drag_current = None
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802, ARG002
        self._set_visible_range(0.0, 1.0)

    def _set_visible_range(self, left: float, right: float) -> None:
        left = _clamp(left, 0.0, 1.0)
        right = _clamp(right, 0.0, 1.0)
        if right - left < 0.08:
            center = (left + right) / 2
            left = center - 0.04
            right = center + 0.04
        if left < 0:
            right -= left
            left = 0.0
        if right > 1:
            left -= right - 1
            right = 1.0
        self._visible_left = _clamp(left, 0.0, 1.0)
        self._visible_right = _clamp(right, 0.0, 1.0)
        self.update()


@dataclass(frozen=True)
class _Page:
    button: QPushButton
    widget: QWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("A股相似阶段搜集")
        self.resize(1280, 820)
        self._workers: list[TaskWorker] = []
        self._pages: list[_Page] = []
        self._full_daily_pause_event: threading.Event | None = None

        root = QWidget()
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_sidebar())
        shell.addWidget(self._build_content(), stretch=1)
        self.setCentralWidget(root)
        self.setStyleSheet(_style_sheet())

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(228)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(22, 24, 18, 24)
        layout.setSpacing(14)

        title = QLabel("A股相似阶段")
        title.setObjectName("brandTitle")
        subtitle = QLabel("研究工作台")
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(18)

        self.nav_overview = _nav_button("总览")
        self.nav_history = _nav_button("历史时序")
        self.nav_cross = _nav_button("横截面")
        self.nav_review = _nav_button("走势复盘")
        self.nav_data = _nav_button("数据管理")
        for button in (self.nav_overview, self.nav_history, self.nav_cross, self.nav_review, self.nav_data):
            layout.addWidget(button)

        layout.addStretch(1)
        api_hint = QLabel("数据 API\nakshare 自动抓取\nTDX 读取本机 tqcenter")
        api_hint.setObjectName("apiHint")
        layout.addWidget(api_hint)
        return sidebar

    def _build_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("content")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 26, 30, 26)
        layout.setSpacing(18)
        layout.addWidget(self._build_header())

        self.stack = QStackedWidget()
        self._add_page(self.nav_overview, self._build_overview_page())
        self._add_page(self.nav_history, self._build_history_page())
        self._add_page(self.nav_cross, self._build_cross_section_page())
        self._add_page(self.nav_review, self._build_review_page())
        self._add_page(self.nav_data, self._build_data_page())
        layout.addWidget(self.stack, stretch=1)
        self._select_page(0)
        return content

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("header")
        layout = QGridLayout(header)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.data_root_input = QLineEdit(str(default_desktop_data_root()))
        self.timeframe_input = QComboBox()
        self.timeframe_input.addItems(["1d", "30m", "15m", "5m", "1m"])
        self.adjust_input = QComboBox()
        self.adjust_input.addItems(["qfq", "hfq", ""])
        self.data_mode_input = QComboBox()
        self.data_mode_input.addItems(["本地 parquet", "数据 API"])
        self.data_mode_input.setMinimumWidth(150)
        self.data_source_input = QComboBox()
        self.data_source_input.addItems(["local", "akshare", "tdx"])
        self.data_source_input.setMinimumWidth(120)
        self.api_url_input = QLineEdit("http://127.0.0.1:8765")
        self.api_url_input.setMinimumWidth(210)
        self.api_url_input.setCursorPosition(0)
        self.tdx_path_input = QLineEdit("")
        self.tdx_path_input.setMinimumWidth(210)
        self.tdx_path_input.setPlaceholderText("通达信安装目录或 PYPlugins/user")
        self.tdx_path_browse_button = _form_button("选择TDX目录", maximum_width=142)
        self.tdx_path_browse_button.clicked.connect(lambda: self._choose_tdx_directory_into(self.tdx_path_input))
        self.data_mode_input.currentTextChanged.connect(lambda _: self._apply_header_visibility())
        self.data_source_input.currentTextChanged.connect(lambda _: self._apply_header_visibility())
        browse = _form_button("选择目录", maximum_width=118)
        browse.clicked.connect(self._choose_data_root)

        self.data_root_label = _field_label("本地行情根目录")
        self.timeframe_label = _field_label("周期")
        self.adjust_label = _field_label("复权")
        self.data_mode_label = _field_label("数据模式")
        self.data_source_label = _field_label("数据源")
        self.api_url_label = _field_label("API 地址")
        self.tdx_path_label = _field_label("TDX 路径")

        layout.addWidget(self.data_root_label, 0, 0)
        layout.addWidget(self.data_root_input, 1, 0)
        layout.addWidget(browse, 1, 1)
        layout.addWidget(self.timeframe_label, 0, 2)
        layout.addWidget(self.timeframe_input, 1, 2)
        layout.addWidget(self.adjust_label, 0, 3)
        layout.addWidget(self.adjust_input, 1, 3)
        layout.addWidget(self.data_mode_label, 0, 4)
        layout.addWidget(self.data_mode_input, 1, 4)
        layout.addWidget(self.data_source_label, 2, 0)
        layout.addWidget(self.data_source_input, 3, 0)
        layout.addWidget(self.api_url_label, 2, 1)
        layout.addWidget(self.api_url_input, 3, 1, 1, 2)
        layout.addWidget(self.tdx_path_label, 2, 3)
        layout.addWidget(self.tdx_path_input, 3, 3, 1, 2)
        layout.addWidget(self.tdx_path_browse_button, 3, 5)
        layout.setColumnStretch(0, 5)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)
        layout.setColumnStretch(3, 2)
        layout.setColumnStretch(4, 2)
        layout.setColumnStretch(5, 0)
        self._apply_header_visibility()
        return header

    def _build_overview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(_page_title("总览", "用统一桌面入口承载历史相似、横截面相似和走势复盘。"))

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        self.symbol_count_label = QLabel("尚未读取")
        cards.addWidget(_metric_card("本地标的", self.symbol_count_label, "读取 parquet 目录下可用代码"), 0, 0)
        cards.addWidget(
            _metric_card(
                "数据 API",
                QLabel("127.0.0.1:8765"),
                "akshare 自动抓取；tdx 需要本机通达信",
            ),
            0,
            1,
        )
        cards.addWidget(_metric_card("界面原则", QLabel("简洁 / 可扩展"), "Qt 层只调服务，不复制算法"), 0, 2)
        layout.addLayout(cards)

        layout.addWidget(_section_label("核心模块"))
        module_cards = QGridLayout()
        module_cards.setHorizontalSpacing(14)
        module_cards.setVerticalSpacing(14)
        for column, module in enumerate(overview_modules()):
            module_cards.addWidget(_module_card(module, self._select_page), 0, column)
        layout.addLayout(module_cards)

        refresh = QPushButton("刷新本地标的")
        refresh.setObjectName("primaryButton")
        refresh.clicked.connect(self._refresh_symbols)
        layout.addWidget(refresh, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return page

    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(_page_title("历史时序相似", "搜索同一标的过去最像当前窗口的阶段。"))

        form = QGridLayout()
        self.history_symbol = _symbol_text("399006.SZ")
        self.history_start = _date_edit(QDate.currentDate().addMonths(-3))
        self.history_as_of = _date_edit(QDate.currentDate())
        self.history_window = _spin(2, 240, 20)
        self.history_top_n = _spin(1, 50, 10)
        self.history_forward_windows = QLineEdit("5,20,60")
        self.history_candidate_n = _spin(10, 1000, 100)
        self.history_exclusion = _spin(0, 500, 20)
        self.history_gap = _spin(0, 365, 20)
        self.history_path_weight = _double_spin(0.0, 1.0, 0.7, 0.05)
        self.history_algorithm = _algorithm_combo(mode="history")
        self.history_button = QPushButton("运行搜索")
        self.history_button.setObjectName("primaryButton")
        self.history_button.clicked.connect(self._run_history)
        self.history_import_button = QPushButton("导入代码")
        self.history_import_button.clicked.connect(lambda: self._import_symbols_into(self.history_symbol))
        self.history_export_button = QPushButton("导出 CSV")
        self.history_export_button.clicked.connect(lambda: self._export_table(self.history_table, "history_similarity.csv"))
        self.history_status = QLabel("等待运行")
        self.history_status.setObjectName("mutedText")
        _add_form_row(form, 0, "目标代码", self.history_symbol, "开始日期", self.history_start)
        _add_form_row(form, 1, "结束日期", self.history_as_of, "快捷长度", self.history_window)
        form.addWidget(_field_label("展示数量"), 2, 0)
        form.addWidget(self.history_top_n, 2, 1)
        form.addWidget(_field_label("后验窗口"), 2, 2)
        form.addWidget(self.history_forward_windows, 2, 3)
        _add_form_row(form, 3, "初筛候选", self.history_candidate_n, "排除近邻K线", self.history_exclusion)
        _add_form_row(form, 4, "样本间隔天数", self.history_gap, "走势权重", self.history_path_weight)
        form.addWidget(_field_label("相似算法"), 5, 0)
        form.addWidget(self.history_algorithm, 5, 1, 1, 3)
        form.addLayout(
            self._quick_window_buttons(
                self.history_symbol,
                self.history_start,
                self.history_as_of,
                self.history_window,
                self.history_status,
                include_latest_close=True,
            ),
            6,
            0,
            1,
            4,
        )
        self.history_download_start = QLineEdit("2018-01-01")
        self.history_download_button = QPushButton("下载/更新该标的行情")
        self.history_download_button.clicked.connect(self._run_history_download)
        form.addWidget(_field_label("下载开始"), 7, 0)
        form.addWidget(self.history_download_start, 7, 1)
        form.addWidget(self.history_download_button, 7, 2)
        form.addWidget(self.history_button, 8, 0)
        form.addWidget(self.history_import_button, 8, 1)
        form.addWidget(self.history_export_button, 8, 2)
        layout.addLayout(form)

        self.history_chart = LineChartWidget("走势对比")
        self.history_size_spread_chart = LineChartWidget("大小盘价差率")
        self.history_kline_chart = CandlestickChartWidget("K线走势核验")
        self.history_kline_layout = _kline_layout_combo(self.history_kline_chart)
        self.history_kline_toolbar = _kline_toolbar(self.history_kline_layout, self.history_kline_chart)
        self.history_kline_expand_button = self.history_kline_toolbar.findChild(QPushButton, "klineExpandButton")
        self.history_forward_stats_table = _compact_table()
        self.history_bucket_stats_table = _compact_table()
        self.history_size_spread_table = _compact_table()
        self.history_data_table = _compact_table()
        self.history_table = _table()
        self.history_table_expand_button = QPushButton("展开表格")
        self.history_table_expand_button.clicked.connect(lambda: self._open_table_dialog(self.history_table, "历史时序结果"))
        self.history_result_tabs = QTabWidget()
        self.history_result_tabs.addTab(
            _chart_tab(
                (
                    self.history_chart,
                    self.history_size_spread_chart,
                    self.history_kline_toolbar,
                    self.history_kline_chart,
                )
            ),
            "图表",
        )
        self.history_result_tabs.addTab(
            _multi_table_tab(
                (
                    ("后验观察统计", self.history_forward_stats_table),
                    ("相似度分层表现", self.history_bucket_stats_table),
                    ("大小盘价差率", self.history_size_spread_table),
                )
            ),
            "统计",
        )
        self.history_result_tabs.addTab(_multi_table_tab((("本地数据覆盖", self.history_data_table),)), "数据")
        self.history_result_tabs.addTab(_table_tab(self.history_table, self.history_table_expand_button), "表格")
        layout.addWidget(self.history_status)
        layout.addWidget(self.history_result_tabs, stretch=1)
        return page

    def _build_cross_section_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(_page_title("横截面相似", "在同一时间附近找到走势最像目标的标的。"))

        form = QGridLayout()
        self.cross_target = _symbol_text("300750.SZ")
        self.cross_start = _date_edit(QDate.currentDate().addMonths(-3))
        self.cross_end = _date_edit(QDate.currentDate())
        self.cross_top_n = _spin(1, 100, 20)
        self.cross_tolerance = _spin(0, 30, 5)
        self.cross_min_coverage = _double_spin(0.5, 1.0, 0.8, 0.05)
        self.cross_path_weight = _double_spin(0.0, 1.0, 0.7, 0.05)
        self.cross_algorithm = _algorithm_combo(mode="cross")
        self.cross_symbols = QPlainTextEdit("000001.SZ, 600519.SH")
        self.cross_symbols.setFixedHeight(78)
        self.cross_index = QLineEdit("")
        self.cross_index.setPlaceholderText("指数成分，如 000300")
        self.cross_industry = QLineEdit("")
        self.cross_industry.setPlaceholderText("行业板块，如 半导体")
        self.cross_concept = QLineEdit("")
        self.cross_concept.setPlaceholderText("概念板块，如 融资融券")
        self.cross_button = QPushButton("运行搜索")
        self.cross_button.setObjectName("primaryButton")
        self.cross_button.clicked.connect(self._run_cross_section)
        self.cross_target_import_button = QPushButton("导入目标")
        self.cross_target_import_button.clicked.connect(lambda: self._import_symbols_into(self.cross_target))
        self.cross_universe_import_button = QPushButton("导入候选")
        self.cross_universe_import_button.clicked.connect(lambda: self._import_symbols_into(self.cross_symbols))
        self.cross_export_button = QPushButton("导出 CSV")
        self.cross_export_button.clicked.connect(lambda: self._export_table(self.cross_table, "cross_section_similarity.csv"))
        self.cross_check_button = QPushButton("检查本地数据覆盖")
        self.cross_check_button.clicked.connect(self._run_cross_check)
        _add_form_row(form, 0, "目标代码", self.cross_target, "开始日期", self.cross_start)
        _add_form_row(form, 1, "结束日期", self.cross_end, "展示数量", self.cross_top_n)
        form.addWidget(_field_label("日期容错"), 2, 0)
        form.addWidget(self.cross_tolerance, 2, 1)
        form.addWidget(_field_label("最小覆盖率"), 2, 2)
        form.addWidget(self.cross_min_coverage, 2, 3)
        _add_form_row(form, 3, "走势权重", self.cross_path_weight, "相似算法", self.cross_algorithm)
        form.addWidget(_field_label("候选范围"), 4, 0)
        form.addWidget(self.cross_symbols, 4, 1, 1, 3)
        form.addWidget(_field_label("指数成分"), 5, 0)
        form.addWidget(self.cross_index, 5, 1)
        form.addWidget(_field_label("行业/概念"), 5, 2)
        group_row = QHBoxLayout()
        group_row.setSpacing(8)
        group_row.addWidget(self.cross_industry)
        group_row.addWidget(self.cross_concept)
        form.addLayout(group_row, 5, 3)
        form.addLayout(
            self._quick_window_buttons(self.cross_target, self.cross_start, self.cross_end, None, None),
            6,
            0,
            1,
            4,
        )
        self.cross_download_button = QPushButton("检查并下载缺失行情")
        self.cross_download_button.clicked.connect(self._run_cross_download_missing)
        form.addWidget(self.cross_check_button, 7, 0, 1, 2)
        form.addWidget(self.cross_download_button, 7, 2, 1, 2)
        form.addWidget(self.cross_button, 8, 0)
        form.addWidget(self.cross_target_import_button, 8, 1)
        form.addWidget(self.cross_universe_import_button, 8, 2)
        form.addWidget(self.cross_export_button, 8, 3)
        layout.addLayout(form)

        self.cross_status = QLabel("等待运行")
        self.cross_status.setObjectName("mutedText")
        self.cross_chart = BarChartWidget("Top 相似度")
        self.cross_data_chart = BarChartWidget("本地覆盖状态")
        self.cross_kline_chart = CandlestickChartWidget("K线走势核验")
        self.cross_kline_layout = _kline_layout_combo(self.cross_kline_chart)
        self.cross_kline_toolbar = _kline_toolbar(self.cross_kline_layout, self.cross_kline_chart)
        self.cross_kline_expand_button = self.cross_kline_toolbar.findChild(QPushButton, "klineExpandButton")
        self.cross_forward_stats_table = _compact_table()
        self.cross_bucket_stats_table = _compact_table()
        self.cross_skipped_table = _compact_table()
        self.cross_data_table = _compact_table()
        self.cross_table = _table()
        self.cross_table_expand_button = QPushButton("展开表格")
        self.cross_table_expand_button.clicked.connect(lambda: self._open_table_dialog(self.cross_table, "横截面结果"))
        self.cross_result_tabs = QTabWidget()
        self.cross_result_tabs.addTab(
            _chart_tab(
                (
                    self.cross_chart,
                    self.cross_kline_toolbar,
                    self.cross_kline_chart,
                )
            ),
            "图表",
        )
        self.cross_result_tabs.addTab(
            _multi_table_tab(
                (
                    ("后验观察统计", self.cross_forward_stats_table),
                    ("相似度分层表现", self.cross_bucket_stats_table),
                    ("跳过样本", self.cross_skipped_table),
                )
            ),
            "统计",
        )
        self.cross_result_tabs.addTab(_chart_tab((self.cross_data_chart, self.cross_data_table)), "数据")
        self.cross_result_tabs.addTab(_table_tab(self.cross_table, self.cross_table_expand_button), "表格")
        layout.addWidget(self.cross_status)
        layout.addWidget(self.cross_result_tabs, stretch=1)
        return page

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(_page_title("走势复盘", "把单段行情压缩成收益、回撤、波段和交易难度。"))

        form = QGridLayout()
        self.review_symbol = _symbol_text("601888.SH, 300750.SZ")
        self.review_start = _date_edit(QDate.currentDate().addMonths(-3))
        self.review_end = _date_edit(QDate.currentDate())
        self.review_min_swing = _double_spin(1.0, 30.0, 5.0, 1.0)
        self.review_min_segment = _spin(2, 60, 3)
        self.review_max_segments = _spin(1, 20, 6)
        self.review_index_symbols = QLineEdit("000300.SH, 000852.SH, 399006.SZ")
        self.review_proxy_symbols = QLineEdit("")
        self.review_proxy_symbols.setPlaceholderText("板块/ETF/指数代理代码")
        self.review_industry = QLineEdit("")
        self.review_industry.setPlaceholderText("行业名称")
        self.review_concept = QLineEdit("")
        self.review_concept.setPlaceholderText("概念名称")
        self.review_sector_coverage = _double_spin(0.3, 1.0, 0.5, 0.05)
        self.review_load_etf_button = QPushButton("加载主要ETF")
        self.review_load_etf_button.clicked.connect(self._load_review_popular_etfs)
        self.review_add_etf_button = QPushButton("加入选中ETF")
        self.review_add_etf_button.clicked.connect(self._append_selected_review_etfs)
        self.review_source = QComboBox()
        self.review_source.addItems(["默认复盘", "AI 复盘"])
        self.review_source.currentTextChanged.connect(lambda _: self._apply_review_ai_visibility())
        self.review_ai_provider = QComboBox()
        self.review_ai_provider.addItems([preset.label for preset in provider_presets().values()])
        self.review_ai_provider.currentTextChanged.connect(lambda _: self._apply_review_ai_defaults())
        self.review_ai_model = QLineEdit()
        self.review_ai_base_url = QLineEdit()
        self.review_ai_api_key = QLineEdit()
        self.review_ai_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.review_ai_thinking = QCheckBox("Thinking")
        self.review_ai_provider_label = _field_label("AI 供应商")
        self.review_ai_model_label = _field_label("模型")
        self.review_ai_base_url_label = _field_label("Base URL")
        self.review_ai_api_key_label = _field_label("API Key")
        self._apply_review_ai_defaults()
        self._apply_review_ai_visibility()
        self.review_button = QPushButton("生成复盘")
        self.review_button.setObjectName("primaryButton")
        self.review_button.clicked.connect(self._run_review)
        self.review_import_button = QPushButton("导入代码")
        self.review_import_button.clicked.connect(lambda: self._import_symbols_into(self.review_symbol))
        self.review_export_button = QPushButton("导出 CSV")
        self.review_export_button.clicked.connect(lambda: self._export_table(self.review_table, "review_segments.csv"))
        _add_form_row(form, 0, "目标代码", self.review_symbol, "开始日期", self.review_start)
        form.addWidget(_field_label("结束日期"), 1, 0)
        form.addWidget(self.review_end, 1, 1)
        form.addWidget(_field_label("复盘来源"), 1, 2)
        form.addWidget(self.review_source, 1, 3)
        _add_form_row(form, 2, "最小波段幅度(%)", self.review_min_swing, "最小段落K线", self.review_min_segment)
        form.addWidget(_field_label("最大波段数"), 3, 0)
        form.addWidget(self.review_max_segments, 3, 1)
        form.addWidget(_field_label("指数对比"), 3, 2)
        form.addWidget(self.review_index_symbols, 3, 3)
        _add_form_row(form, 4, "代理代码", self.review_proxy_symbols, "行业名称", self.review_industry)
        form.addWidget(_field_label("概念名称"), 5, 0)
        form.addWidget(self.review_concept, 5, 1)
        form.addWidget(_field_label("成分覆盖率"), 5, 2)
        form.addWidget(self.review_sector_coverage, 5, 3)
        form.addWidget(self.review_load_etf_button, 6, 0)
        form.addWidget(self.review_add_etf_button, 6, 1)
        form.addWidget(self.review_ai_provider_label, 7, 0)
        form.addWidget(self.review_ai_provider, 7, 1)
        form.addWidget(self.review_ai_model_label, 7, 2)
        form.addWidget(self.review_ai_model, 7, 3)
        form.addWidget(self.review_ai_base_url_label, 8, 0)
        form.addWidget(self.review_ai_base_url, 8, 1)
        form.addWidget(self.review_ai_api_key_label, 8, 2)
        form.addWidget(self.review_ai_api_key, 8, 3)
        form.addWidget(self.review_ai_thinking, 9, 0)
        form.addLayout(
            self._quick_window_buttons(self.review_symbol, self.review_start, self.review_end, None, None),
            10,
            0,
            1,
            4,
        )
        form.addWidget(self.review_button, 11, 0)
        form.addWidget(self.review_import_button, 11, 1)
        form.addWidget(self.review_export_button, 11, 2)
        layout.addLayout(form)

        self.review_summary = QLabel("等待运行")
        self.review_summary.setObjectName("summaryText")
        self.review_summary.setWordWrap(True)
        self.review_chart = CandlestickChartWidget("复盘K线")
        self.review_relative_chart = LineChartWidget("相对走势")
        self.review_kline_layout = _kline_layout_combo(self.review_chart)
        self.review_kline_toolbar = _kline_toolbar(self.review_kline_layout, self.review_chart)
        self.review_kline_expand_button = self.review_kline_toolbar.findChild(QPushButton, "klineExpandButton")
        self.review_cards = ReviewCardsWidget()
        self.review_text = QTextEdit()
        self.review_text.setObjectName("reviewText")
        self.review_text.setReadOnly(True)
        self.review_text.setMinimumHeight(170)
        self.review_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.review_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.review_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.review_text_toggle_button = QPushButton("收起文本")
        self.review_text_toggle_button.setObjectName("flatActionButton")
        self.review_text_toggle_button.clicked.connect(
            lambda: _toggle_panel_widget(self.review_text, self.review_text_toggle_button, "展开文本", "收起文本")
        )
        self.review_text_panel = _text_output_panel(self.review_text, self.review_text_toggle_button)
        self.review_table = _table()
        self.review_table_expand_button = QPushButton("展开表格")
        self.review_table_expand_button.clicked.connect(lambda: self._open_table_dialog(self.review_table, "走势复盘波段"))
        self.review_overview_table = _compact_table()
        self.review_ranking_table = _compact_table()
        self.review_comparison_table = _compact_table()
        self.review_script_table = _compact_table()
        self.review_auto_etf_table = _compact_table()
        self.review_popular_etf_table = _compact_table()
        self.review_popular_etf_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.review_result_tabs = QTabWidget()
        self.review_result_tabs.addTab(
            _chart_tab(
                (
                    self.review_cards,
                    self.review_kline_toolbar,
                    self.review_chart,
                    self.review_relative_chart,
                    self.review_text_panel,
                )
            ),
            "图表/锐评",
        )
        self.review_result_tabs.addTab(
            _multi_table_tab(
                (
                    ("排序总表", self.review_ranking_table),
                    ("区间概览", self.review_overview_table),
                    ("指数 / 板块对比", self.review_comparison_table),
                    ("视频脚本视角", self.review_script_table),
                    ("ETF自动代理", self.review_auto_etf_table),
                    ("主要ETF", self.review_popular_etf_table),
                )
            ),
            "统计",
        )
        self.review_result_tabs.addTab(_table_tab(self.review_table, self.review_table_expand_button), "表格")
        layout.addWidget(self.review_summary)
        layout.addWidget(self.review_result_tabs, stretch=1)
        return page

    def _build_data_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(_page_title("数据管理", "检查本地 K 线覆盖，预览并执行文件迁移。"))

        self.data_root_hint = QLabel("")
        self.data_root_hint.setObjectName("dataRootHint")
        self.data_root_hint.setWordWrap(True)
        layout.addWidget(self.data_root_hint)
        self.data_root_input.textChanged.connect(lambda _: self._refresh_data_root_hint())
        self.timeframe_input.currentTextChanged.connect(lambda _: self._refresh_data_root_hint())
        self.adjust_input.currentTextChanged.connect(lambda _: self._refresh_data_root_hint())
        self._refresh_data_root_hint()

        self.benchmark_command = QPlainTextEdit()
        self.benchmark_command.setReadOnly(True)
        self.benchmark_command.setMaximumHeight(92)
        self.benchmark_command.setPlainText(
            _benchmark_command(
                self.data_root_input.text().strip(),
                self.timeframe_input.currentText(),
                self.adjust_input.currentText(),
            )
        )
        layout.addWidget(_section_label("算法核验图集"))
        layout.addWidget(self.benchmark_command)

        coverage_form = QGridLayout()
        self.data_symbols = _symbol_text("000852.SH, 000300.SH")
        self.data_start = _date_edit(QDate.currentDate().addMonths(-3))
        self.data_end = _date_edit(QDate.currentDate())
        self.data_check_button = _form_button("检查覆盖", primary=True)
        self.data_check_button.setObjectName("primaryButton")
        self.data_check_button.clicked.connect(self._run_data_check)
        self.data_download_engine = QComboBox()
        self.data_download_engine.addItems(["akshare", "tdx", "openbb", "trend"])
        self.data_download_provider = QLineEdit("")
        self.data_download_provider.setPlaceholderText("TDX: 通达信目录或 PYPlugins/user；OpenBB: akshare")
        self.data_tdx_provider_browse_button = _form_button("选择TDX目录", maximum_width=142)
        self.data_tdx_provider_browse_button.clicked.connect(lambda: self._choose_tdx_directory_into(self.data_download_provider))
        self.data_update_button = _form_button("下载/更新行情", primary=True)
        self.data_update_button.setObjectName("primaryButton")
        self.data_update_button.clicked.connect(self._run_data_update)
        self.data_export_button = _form_button("导出覆盖 CSV")
        self.data_export_button.clicked.connect(lambda: self._export_table(self.data_coverage_table, "data_coverage.csv"))
        self.data_coverage_summary = QLabel("0/0 可用")
        self.data_coverage_summary.setObjectName("coverageSummary")
        self.data_coverage_detail_button = _form_button("查看明细")
        self.data_coverage_detail_button.setEnabled(False)
        self.data_coverage_detail_button.clicked.connect(self._open_data_coverage_detail)
        self.data_update_progress_bar = _progress_bar()
        _add_form_row(coverage_form, 0, "检查代码", self.data_symbols, "开始日期", self.data_start)
        coverage_form.addWidget(_field_label("结束日期"), 1, 0)
        coverage_form.addWidget(self.data_end, 1, 1)
        coverage_form.addWidget(_field_label("下载引擎"), 1, 2)
        coverage_form.addWidget(self.data_download_engine, 1, 3)
        coverage_form.addWidget(_field_label("Provider/路径"), 2, 0)
        coverage_form.addWidget(self.data_download_provider, 2, 1, 1, 3)
        coverage_form.addWidget(self.data_tdx_provider_browse_button, 2, 4)
        coverage_form.addWidget(self.data_check_button, 3, 0)
        coverage_form.addWidget(self.data_update_button, 3, 1)
        coverage_form.addWidget(self.data_export_button, 3, 2)
        coverage_form.addWidget(self.data_coverage_summary, 4, 0, 1, 2)
        coverage_form.addWidget(self.data_coverage_detail_button, 4, 2)
        coverage_form.addWidget(self.data_update_progress_bar, 4, 3, 1, 2)
        coverage_form.setColumnStretch(4, 0)
        layout.addWidget(
            _data_operation_panel(
                "覆盖与补数据",
                "检查本地 parquet 覆盖；按 AkShare / TDX / OpenBB / trend 下载或修补缺失区间。",
                coverage_form,
            )
        )

        self.data_status = QLabel("等待检查")
        self.data_status.setObjectName("mutedText")
        self.data_coverage_chart = BarChartWidget("覆盖状态")
        self.data_coverage_table = _table()
        self._data_coverage_detail_frame = pd.DataFrame()

        price_import_form = QGridLayout()
        self.price_import_file = QLineEdit("")
        self.price_import_file.setPlaceholderText("CSV/Parquet 价格数据文件")
        self.price_import_fallback_symbol = QLineEdit("")
        self.price_import_fallback_symbol.setPlaceholderText("文件无代码列时填写，如 000001")
        price_file_button = _form_button("选择价格文件", maximum_width=132)
        price_file_button.clicked.connect(self._choose_price_file)
        self.price_import_button = _form_button("导入价格数据", primary=True)
        self.price_import_button.setObjectName("primaryButton")
        self.price_import_button.clicked.connect(self._run_price_import)
        price_import_form.addWidget(_field_label("价格文件"), 0, 0)
        price_import_form.addWidget(self.price_import_file, 0, 1, 1, 3)
        price_import_form.addWidget(price_file_button, 0, 4)
        price_import_form.addWidget(_field_label("默认代码"), 1, 0)
        price_import_form.addWidget(self.price_import_fallback_symbol, 1, 1, 1, 3)
        price_import_form.addWidget(self.price_import_button, 1, 4)
        price_import_form.setColumnStretch(1, 1)
        layout.addWidget(
            _data_operation_panel(
                "自定义价格导入",
                "导入 CSV 或 Parquet，落到当前数据根目录、周期和复权口径。",
                price_import_form,
            )
        )

        self.price_import_status = QLabel("等待导入")
        self.price_import_status.setObjectName("mutedText")
        self.price_import_table = _compact_table()

        full_daily_form = QGridLayout()
        self.full_daily_start = _date_edit(QDate(1990, 1, 1))
        self.full_daily_end = _date_edit(QDate.currentDate())
        self.full_daily_provider = QLineEdit("")
        self.full_daily_provider.setPlaceholderText("留空使用顶部 TDX 路径；也可选择通达信目录")
        self.full_daily_tdx_browse_button = _form_button("选择TDX目录", maximum_width=142)
        self.full_daily_tdx_browse_button.clicked.connect(lambda: self._choose_tdx_directory_into(self.full_daily_provider))
        self.full_daily_batch_size = _spin(1, 500, 100)
        self.full_daily_skip_available = QCheckBox("跳过已覆盖")
        self.full_daily_skip_available.setChecked(True)
        self.full_daily_include_indexes = QCheckBox("包含常用指数")
        self.full_daily_include_indexes.setChecked(True)
        self.full_daily_extra_symbols = QLineEdit("")
        self.full_daily_extra_symbols.setPlaceholderText("额外 ETF/指数，如 159915.SZ")
        self.full_daily_plan_button = _form_button("预览全A日线")
        self.full_daily_plan_button.clicked.connect(self._plan_full_daily_update)
        self.full_daily_run_button = _form_button("开始全A下载", primary=True)
        self.full_daily_run_button.setObjectName("primaryButton")
        self.full_daily_run_button.clicked.connect(self._run_full_daily_update)
        self.full_daily_pause_button = _form_button("暂停下载")
        self.full_daily_pause_button.setEnabled(False)
        self.full_daily_pause_button.clicked.connect(self._toggle_full_daily_pause)
        self.full_daily_summary = QLabel("0/0 待下载")
        self.full_daily_summary.setObjectName("coverageSummary")
        self.full_daily_detail_button = _form_button("查看明细")
        self.full_daily_detail_button.setEnabled(False)
        self.full_daily_detail_button.clicked.connect(self._open_full_daily_detail)
        self.full_daily_progress_bar = _progress_bar()
        full_daily_form.addWidget(_field_label("开始日期"), 0, 0)
        full_daily_form.addWidget(self.full_daily_start, 0, 1)
        full_daily_form.addWidget(_field_label("结束日期"), 0, 2)
        full_daily_form.addWidget(self.full_daily_end, 0, 3)
        full_daily_form.addWidget(_field_label("TDX 路径"), 1, 0)
        full_daily_form.addWidget(self.full_daily_provider, 1, 1, 1, 3)
        full_daily_form.addWidget(self.full_daily_tdx_browse_button, 1, 4)
        full_daily_form.addWidget(_field_label("批次大小"), 2, 0)
        full_daily_form.addWidget(self.full_daily_batch_size, 2, 1)
        full_daily_form.addWidget(self.full_daily_skip_available, 2, 2)
        full_daily_form.addWidget(self.full_daily_include_indexes, 2, 3)
        full_daily_form.addWidget(_field_label("额外代码"), 3, 0)
        full_daily_form.addWidget(self.full_daily_extra_symbols, 3, 1, 1, 3)
        full_daily_form.addWidget(self.full_daily_plan_button, 4, 0)
        full_daily_form.addWidget(self.full_daily_run_button, 4, 1)
        full_daily_form.addWidget(self.full_daily_pause_button, 4, 2)
        full_daily_form.addWidget(self.full_daily_summary, 5, 0, 1, 2)
        full_daily_form.addWidget(self.full_daily_detail_button, 5, 2)
        full_daily_form.addWidget(self.full_daily_progress_bar, 5, 3, 1, 2)
        full_daily_form.setColumnStretch(1, 1)
        full_daily_form.setColumnStretch(3, 1)
        full_daily_form.setColumnStretch(4, 0)
        layout.addWidget(
            _data_operation_panel(
                "全A日线批量更新",
                "使用 TDX 股票清单批次下载日线；先预览覆盖，再按批次执行。",
                full_daily_form,
            )
        )

        self.full_daily_status = QLabel("等待预览")
        self.full_daily_status.setObjectName("mutedText")
        self.full_daily_table = _table()
        self._full_daily_detail_frame = pd.DataFrame()

        migration_form = QGridLayout()
        self.migration_source = QLineEdit("")
        self.migration_destination = QLineEdit("")
        self.migration_mode = QComboBox()
        self.migration_mode.addItems(["copy", "move"])
        self.migration_overwrite = QCheckBox("覆盖同名文件")
        source_file = _form_button("来源文件", maximum_width=108)
        source_file.clicked.connect(lambda: self._choose_file_into(self.migration_source))
        source_folder = _form_button("来源目录", maximum_width=108)
        source_folder.clicked.connect(lambda: self._choose_directory_into(self.migration_source, "选择来源目录"))
        destination_folder = _form_button("目标目录", maximum_width=108)
        destination_folder.clicked.connect(lambda: self._choose_directory_into(self.migration_destination, "选择目标目录"))
        self.migration_preview_button = _form_button("预览迁移")
        self.migration_preview_button.clicked.connect(self._preview_migration)
        self.migration_execute_button = _form_button("执行迁移", primary=True)
        self.migration_execute_button.setObjectName("primaryButton")
        self.migration_execute_button.clicked.connect(self._execute_migration)
        self.migration_export_button = _form_button("导出迁移 CSV")
        self.migration_export_button.clicked.connect(lambda: self._export_table(self.migration_table, "kline_migration.csv"))
        migration_form.addWidget(_field_label("来源"), 0, 0)
        migration_form.addWidget(self.migration_source, 0, 1, 1, 3)
        migration_form.addWidget(source_file, 0, 4)
        migration_form.addWidget(source_folder, 0, 5)
        migration_form.addWidget(_field_label("目标"), 1, 0)
        migration_form.addWidget(self.migration_destination, 1, 1, 1, 3)
        migration_form.addWidget(destination_folder, 1, 4)
        migration_form.addWidget(_field_label("模式"), 2, 0)
        migration_form.addWidget(self.migration_mode, 2, 1)
        migration_form.addWidget(self.migration_overwrite, 2, 2)
        migration_form.addWidget(self.migration_preview_button, 3, 0)
        migration_form.addWidget(self.migration_execute_button, 3, 1)
        migration_form.addWidget(self.migration_export_button, 3, 2)
        migration_form.setColumnStretch(1, 1)
        migration_form.setColumnStretch(2, 1)
        migration_form.setColumnStretch(3, 1)
        layout.addWidget(
            _data_operation_panel(
                "K线文件迁移",
                "先预览 copy/move 计划，再执行文件迁移，避免误操作。",
                migration_form,
            )
        )

        self.migration_status = QLabel("等待预览")
        self.migration_status.setObjectName("mutedText")
        self.migration_table = _table()
        self.data_result_tabs = QTabWidget()
        self.data_result_tabs.setObjectName("dataResultTabs")
        self.data_result_tabs.addTab(
            _chart_tab((self.data_status, self.data_coverage_chart, self.data_coverage_table)),
            "覆盖",
        )
        self.data_result_tabs.addTab(_chart_tab((self.price_import_status, self.price_import_table)), "价格导入")
        self.data_result_tabs.addTab(_chart_tab((self.full_daily_status, self.full_daily_table)), "全A日线")
        self.data_result_tabs.addTab(_chart_tab((self.migration_status, self.migration_table)), "K线迁移")
        layout.addWidget(self.data_result_tabs, stretch=1)
        return page

    def _add_page(self, button: QPushButton, widget: QWidget) -> None:
        index = self.stack.addWidget(_scroll_page(widget))
        button.clicked.connect(lambda checked=False, page_index=index: self._select_page(page_index))
        self._pages.append(_Page(button=button, widget=widget))

    def _select_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for page_index, page in enumerate(self._pages):
            page.button.setProperty("active", page_index == index)
            page.button.style().unpolish(page.button)
            page.button.style().polish(page.button)

    def _service(self) -> DesktopSearchService:
        return DesktopSearchService(
            DesktopAppConfig(
                data_root=self.data_root_input.text().strip(),
                timeframe=self.timeframe_input.currentText(),
                adjust=self.adjust_input.currentText(),
                data_mode="api" if self.data_mode_input.currentText() == "数据 API" else "local",
                data_source=self.data_source_input.currentText(),
                api_base_url=self.api_url_input.text().strip(),
                tdx_path=self.tdx_path_input.text().strip(),
            )
        )

    def _apply_header_visibility(self) -> None:
        visible = set(visible_data_fields(self.data_mode_input.currentText(), self.data_source_input.currentText()))
        field_widgets = {
            "data_source": (self.data_source_label, self.data_source_input),
            "api_url": (self.api_url_label, self.api_url_input),
            "tdx_path": (self.tdx_path_label, self.tdx_path_input, self.tdx_path_browse_button),
        }
        for field, widgets in field_widgets.items():
            should_show = field in visible
            for widget in widgets:
                widget.setVisible(should_show)

    def _refresh_data_root_hint(self) -> None:
        hint = getattr(self, "data_root_hint", None)
        if not isinstance(hint, QLabel):
            return
        root_text = self.data_root_input.text().strip()
        timeframe = self.timeframe_input.currentText()
        adjust = self.adjust_input.currentText()
        if not root_text:
            hint.setText("下载前先确认数据存储根目录；当前为空，下载后检查会继续报 missing_file。")
            return
        write_root = resolve_timeframe_root(Path(root_text).expanduser(), timeframe)
        if adjust:
            write_root = write_root / adjust
        hint.setText(f"下载前先确认数据存储根目录；parquet 将写入：{write_root}")

    def _apply_review_ai_defaults(self) -> None:
        preset = _provider_preset_from_label(self.review_ai_provider.currentText())
        self.review_ai_model.setText(preset.default_model)
        self.review_ai_base_url.setText(preset.default_base_url)
        self.review_ai_thinking.setChecked(bool(preset.supports_thinking))
        self.review_ai_thinking.setEnabled(bool(preset.supports_thinking))

    def _apply_review_ai_visibility(self) -> None:
        should_show = self.review_source.currentText() == "AI 复盘"
        for widget in (
            self.review_ai_provider,
            self.review_ai_provider_label,
            self.review_ai_model,
            self.review_ai_model_label,
            self.review_ai_base_url,
            self.review_ai_base_url_label,
            self.review_ai_api_key,
            self.review_ai_api_key_label,
            self.review_ai_thinking,
        ):
            widget.setVisible(should_show)

    def _choose_data_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择本地行情根目录", self.data_root_input.text())
        if selected:
            self.data_root_input.setText(selected)
            self._refresh_data_root_hint()

    def _choose_tdx_directory_into(self, target: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择通达信目录或 PYPlugins/user",
            target.text() or self.tdx_path_input.text() or self.data_root_input.text(),
        )
        if selected:
            target.setText(selected)

    def _choose_directory_into(self, target: QLineEdit, title: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, title, target.text() or self.data_root_input.text())
        if selected:
            target.setText(selected)

    def _choose_file_into(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择K线文件",
            target.text() or self.data_root_input.text(),
            "K线文件 (*.parquet *.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if path:
            target.setText(path)

    def _choose_price_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择价格数据文件",
            self.price_import_file.text() or self.data_root_input.text(),
            "价格数据文件 (*.csv *.parquet);;所有文件 (*)",
        )
        if path:
            self.price_import_file.setText(path)

    def _import_symbols_into(self, target: QPlainTextEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入代码文件",
            "",
            "代码文件 (*.csv *.xlsx *.xls *.parquet);;所有文件 (*)",
        )
        if not path:
            return
        try:
            symbols = import_symbol_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        target.setPlainText(", ".join(symbols))

    def _quick_window_buttons(
        self,
        symbol_input: QPlainTextEdit,
        start_edit: DatePicker,
        end_edit: DatePicker,
        length_spin: QSpinBox | None,
        status_label: QLabel | None,
        *,
        include_latest_close: bool = False,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(_field_label("快捷区间"))
        if include_latest_close:
            self.history_latest_button = QPushButton("最新收盘")
            self.history_latest_button.clicked.connect(
                lambda checked=False: self._apply_latest_close(symbol_input, end_edit, status_label)
            )
            row.addWidget(self.history_latest_button)
        for window_size in (5, 10, 20, 60, 120):
            button = QPushButton(f"近{window_size}根")
            button.clicked.connect(
                lambda checked=False, size=window_size: self._apply_quick_window(
                    symbol_input,
                    start_edit,
                    end_edit,
                    size,
                    length_spin=length_spin,
                    status_label=status_label,
                )
            )
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _apply_quick_window(
        self,
        symbol_input: QPlainTextEdit,
        start_edit: DatePicker,
        end_edit: DatePicker,
        window_size: int,
        *,
        length_spin: QSpinBox | None,
        status_label: QLabel | None,
    ) -> None:
        try:
            quick = self._service().quick_window(symbol_input.toPlainText(), window_size=window_size)
        except Exception as exc:  # noqa: BLE001
            if status_label is not None:
                status_label.setText(f"失败：{exc}")
            QMessageBox.warning(self, "快捷区间失败", str(exc))
            return
        start_edit.setDate(QDate.fromString(quick.start, "yyyy-MM-dd"))
        end_edit.setDate(QDate.fromString(quick.end, "yyyy-MM-dd"))
        if length_spin is not None and quick.rows >= length_spin.minimum():
            length_spin.setValue(min(quick.rows, length_spin.maximum()))
        if status_label is not None:
            status_label.setText(quick.message)

    def _apply_latest_close(
        self,
        symbol_input: QPlainTextEdit,
        end_edit: DatePicker,
        status_label: QLabel | None,
    ) -> None:
        try:
            latest = self._service().latest_close(symbol_input.toPlainText())
        except Exception as exc:  # noqa: BLE001
            if status_label is not None:
                status_label.setText(f"失败：{exc}")
            QMessageBox.warning(self, "最新收盘失败", str(exc))
            return
        end_edit.setDate(QDate.fromString(latest.end, "yyyy-MM-dd"))
        if status_label is not None:
            status_label.setText(latest.message)

    def _export_table(self, table: QTableView, default_name: str) -> None:
        model = table.model()
        if not isinstance(model, DataFrameModel) or model.frame().empty:
            QMessageBox.information(self, "没有可导出数据", "当前表格为空。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 CSV",
            default_name,
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not path:
            return
        try:
            output = export_frame_csv(model.frame(), path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", f"已导出：{output}")

    def _open_table_dialog(self, table: QTableView, title: str) -> None:
        model = table.model()
        if not isinstance(model, DataFrameModel) or model.frame().empty:
            QMessageBox.information(self, "没有可展开数据", "当前表格为空。")
            return
        self.data_detail_dialog = _dataframe_detail_dialog(model.frame(), title, self)
        self.data_detail_dialog.exec()

    def _open_data_coverage_detail(self) -> None:
        if self._data_coverage_detail_frame.empty:
            QMessageBox.information(self, "没有覆盖明细", "请先检查覆盖。")
            return
        self.data_detail_dialog = _dataframe_detail_dialog(self._data_coverage_detail_frame, "覆盖明细", self)
        self.data_detail_dialog.show()

    def _open_full_daily_detail(self) -> None:
        if self._full_daily_detail_frame.empty:
            QMessageBox.information(self, "没有全A明细", "请先预览或下载全A日线。")
            return
        self.data_detail_dialog = _dataframe_detail_dialog(self._full_daily_detail_frame, "全A日线明细", self)
        self.data_detail_dialog.show()

    def _refresh_symbols(self) -> None:
        self._run_task(
            lambda: self._service().available_symbols(),
            lambda symbols: self.symbol_count_label.setText(f"{len(symbols):,} 个"),
            status_label=self.symbol_count_label,
        )

    def _run_history(self) -> None:
        request = HistoryRequest(
            symbol=self.history_symbol.toPlainText(),
            as_of=_date_text(self.history_as_of),
            window_start=_date_text(self.history_start),
            window_size=self.history_window.value(),
            forward_windows=_parse_int_tuple(self.history_forward_windows.text()),
            candidate_n=self.history_candidate_n.value(),
            top_n=self.history_top_n.value(),
            exclusion_bars=self.history_exclusion.value(),
            nearby_gap_days=self.history_gap.value(),
            path_weight=self.history_path_weight.value(),
            algorithm=str(self.history_algorithm.currentData() or BASELINE_ALGORITHM),
        )
        self._run_task(
            lambda: self._run_history_operation(request),
            self._show_history_result,
            status_label=self.history_status,
            busy_text="正在搜索历史阶段...",
        )

    def _run_history_operation(self, request: HistoryRequest) -> dict[str, object]:
        service = self._service()
        histories = service.search_history_many(request)
        symbols = tuple(history.symbol for history in histories)
        bars = service.load_bars(symbols=symbols, start="1900-01-01", end=_today_text())
        stock_names = service.resolve_stock_names(symbols, bars=bars)
        size_spread_error = ""
        try:
            size_spread_bars = service.load_bars(
                symbols=("000852.SH", "000300.SH"),
                start="2016-01-01",
                end=_today_text(),
            )
        except Exception as exc:  # noqa: BLE001
            size_spread_bars = pd.DataFrame()
            size_spread_error = f"大小盘价差率不可用：{exc}"
        coverage = service.check_coverage(
            DataCoverageRequest(
                symbols=symbols,
                start=request.window_start or request.as_of,
                end=request.as_of,
            )
        )
        return {
            "histories": histories,
            "bars": bars,
            "coverage": coverage,
            "size_spread_bars": size_spread_bars,
            "size_spread_error": size_spread_error,
            "stock_names": stock_names,
        }

    def _run_cross_section(self) -> None:
        request = self._cross_section_request_from_inputs()
        universe_text = self.cross_symbols.toPlainText()
        index_code = self.cross_index.text().strip()
        industry = self.cross_industry.text().strip()
        concept = self.cross_concept.text().strip()
        self._run_task(
            lambda: self._run_cross_section_operation(
                request,
                universe_text=universe_text,
                index_code=index_code,
                industry=industry,
                concept=concept,
            ),
            self._show_cross_section_result,
            status_label=self.cross_status,
            busy_text="正在比较横截面走势...",
        )

    def _cross_section_request_from_inputs(self, *, universe_symbols: tuple[str, ...] = ()) -> CrossSectionRequest:
        return CrossSectionRequest(
            target_symbol=self.cross_target.toPlainText(),
            universe_symbols=universe_symbols,
            start=_date_text(self.cross_start),
            end=_date_text(self.cross_end),
            top_n=self.cross_top_n.value(),
            min_coverage=self.cross_min_coverage.value(),
            path_weight=self.cross_path_weight.value(),
            date_tolerance_bars=self.cross_tolerance.value(),
            algorithm=str(self.cross_algorithm.currentData() or BASELINE_ALGORITHM),
        )

    def _run_cross_check(self) -> None:
        request = self._cross_section_request_from_inputs()
        universe_text = self.cross_symbols.toPlainText()
        index_code = self.cross_index.text().strip()
        industry = self.cross_industry.text().strip()
        concept = self.cross_concept.text().strip()
        self._run_task(
            lambda: self._run_cross_coverage_operation(
                request,
                universe_text=universe_text,
                index_code=index_code,
                industry=industry,
                concept=concept,
            ),
            self._show_cross_coverage_result,
            status_label=self.cross_status,
            busy_text="正在检查本地数据覆盖...",
        )

    def _run_cross_download_missing(self) -> None:
        request = MissingDataUpdateRequest(
            symbols=self._current_cross_symbols(),
            start=_date_text(self.cross_start),
            end=_date_text(self.cross_end),
            download_engine=self._current_download_engine(),
            provider=self._current_download_provider(),
        )
        self._run_task(
            lambda: self._service().update_missing_bars(request),
            self._show_cross_download_result,
            status_label=self.cross_status,
            busy_text="正在检查并下载缺失行情...",
        )

    def _current_cross_symbols(self) -> tuple[str, ...]:
        service = self._service()
        universe_symbols = service.resolve_universe_symbols(
            symbols=self.cross_symbols.toPlainText(),
            index_code=self.cross_index.text().strip(),
            industry=self.cross_industry.text().strip(),
            concept=self.cross_concept.text().strip(),
        )
        return parse_symbol_list([*parse_symbol_list(self.cross_target.toPlainText()), *universe_symbols])

    def _run_cross_section_operation(
        self,
        request: CrossSectionRequest,
        *,
        universe_text: str,
        index_code: str,
        industry: str,
        concept: str,
    ) -> dict[str, object]:
        service = self._service()
        universe_symbols = service.resolve_universe_symbols(
            symbols=universe_text,
            index_code=index_code,
            industry=industry,
            concept=concept,
        )
        cross_sections = service.search_cross_section_many(
            CrossSectionRequest(
                target_symbol=request.target_symbol,
                universe_symbols=universe_symbols,
                start=request.start,
                end=request.end,
                top_n=request.top_n,
                min_coverage=request.min_coverage,
                path_weight=request.path_weight,
                date_tolerance_bars=request.date_tolerance_bars,
                algorithm=request.algorithm,
            )
        )
        chart_symbols: list[str] = []
        name_symbols: list[str] = [*parse_symbol_list(request.target_symbol), *universe_symbols]
        chart_start = pd.Timestamp(request.start)
        for cross_section in cross_sections:
            chart_symbols.append(cross_section.target_symbol)
            name_symbols.append(cross_section.target_symbol)
            if not cross_section.results.empty and "symbol" in cross_section.results.columns:
                result_symbols = [str(symbol) for symbol in cross_section.results["symbol"].tolist()]
                name_symbols.extend(result_symbols)
                chart_symbols.extend(result_symbols[:6])
            if not cross_section.results.empty and "区间开始" in cross_section.results.columns:
                starts = pd.to_datetime(cross_section.results["区间开始"], errors="coerce").dropna()
                if not starts.empty:
                    chart_start = min(chart_start, starts.min())
        bars = (
            service.load_bars(
                symbols=parse_symbol_list(chart_symbols),
                start=chart_start.strftime("%Y-%m-%d"),
                end=_today_text(),
            )
            if chart_symbols
            else pd.DataFrame()
        )
        stock_names = service.resolve_stock_names(parse_symbol_list(name_symbols), bars=bars)
        coverage = service.cross_section_coverage(
            CrossSectionRequest(
                target_symbol=request.target_symbol,
                universe_symbols=universe_symbols,
                start=request.start,
                end=request.end,
                top_n=request.top_n,
                min_coverage=request.min_coverage,
                path_weight=request.path_weight,
                date_tolerance_bars=request.date_tolerance_bars,
                algorithm=request.algorithm,
            )
        )
        return {"cross_sections": cross_sections, "bars": bars, "coverage": coverage, "stock_names": stock_names}

    def _run_cross_coverage_operation(
        self,
        request: CrossSectionRequest,
        *,
        universe_text: str,
        index_code: str,
        industry: str,
        concept: str,
    ) -> pd.DataFrame:
        service = self._service()
        universe_symbols = service.resolve_universe_symbols(
            symbols=universe_text,
            index_code=index_code,
            industry=industry,
            concept=concept,
        )
        return service.cross_section_coverage(
            CrossSectionRequest(
                target_symbol=request.target_symbol,
                universe_symbols=universe_symbols,
                start=request.start,
                end=request.end,
                top_n=request.top_n,
                min_coverage=request.min_coverage,
                path_weight=request.path_weight,
                date_tolerance_bars=request.date_tolerance_bars,
                algorithm=request.algorithm,
            )
        )

    def _run_review(self) -> None:
        request = ReviewRequest(
            symbol=self.review_symbol.toPlainText(),
            start=_date_text(self.review_start),
            end=_date_text(self.review_end),
            min_swing_return=self.review_min_swing.value() / 100.0,
            min_segment_bars=self.review_min_segment.value(),
            max_segments=self.review_max_segments.value(),
            index_symbols=parse_symbol_list(self.review_index_symbols.text()),
            proxy_symbols=parse_symbol_list(self.review_proxy_symbols.text()),
            industry_name=self.review_industry.text().strip(),
            concept_name=self.review_concept.text().strip(),
            sector_min_coverage=self.review_sector_coverage.value(),
        )
        ai_config = self._review_ai_config() if self.review_source.currentText() == "AI 复盘" else None
        self._run_task(
            lambda: self._run_review_operation(request, ai_config),
            self._show_review_result,
            status_label=self.review_summary,
            busy_text="正在生成复盘...",
        )

    def _load_review_popular_etfs(self) -> None:
        try:
            frame = self._service().popular_review_etfs(limit=10)
        except Exception as exc:  # noqa: BLE001
            self.review_summary.setText(f"ETF候选加载失败：{exc}")
            QMessageBox.warning(self, "ETF候选加载失败", str(exc))
            return
        self._show_review_etf_candidates(frame)
        self.review_summary.setText(f"已加载主要 ETF {len(frame)} 个。")

    def _show_review_etf_candidates(self, frame: pd.DataFrame) -> None:
        self.review_popular_etf_table.setModel(DataFrameModel(frame))
        self.review_popular_etf_table.resizeColumnsToContents()
        dialog = QDialog(self)
        dialog.setObjectName("reviewEtfDialog")
        dialog.setWindowTitle("主要ETF候选")
        dialog.setModal(False)
        dialog.resize(760, 460)

        title = QLabel(f"主要 ETF 候选（{len(frame)} 个）")
        title.setObjectName("summaryText")
        table = _compact_table()
        table.setObjectName("reviewEtfPopupTable")
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setModel(DataFrameModel(frame))
        table.resizeColumnsToContents()

        add_button = QPushButton("加入选中ETF")
        add_button.setObjectName("primaryButton")
        add_button.clicked.connect(lambda: self._append_review_etfs_from_table(table))
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(add_button)
        buttons.addWidget(close_button)

        layout = QVBoxLayout(dialog)
        layout.addWidget(title)
        layout.addWidget(table, stretch=1)
        layout.addLayout(buttons)
        self.review_etf_dialog = dialog
        dialog.show()

    def _append_selected_review_etfs(self) -> None:
        self._append_review_etfs_from_table(self.review_popular_etf_table)

    def _append_review_etfs_from_table(self, table: QTableView) -> None:
        model = table.model()
        if not isinstance(model, DataFrameModel):
            QMessageBox.information(self, "没有ETF候选", "请先加载主要ETF。")
            return
        frame = model.frame()
        selection = table.selectionModel()
        rows = sorted({index.row() for index in selection.selectedRows()}) if selection is not None else []
        symbols = [str(frame.iloc[row].get("symbol", "")).strip() for row in rows if 0 <= row < len(frame)]
        symbols = list(parse_symbol_list(symbols)) if symbols else []
        if not symbols:
            QMessageBox.information(self, "未选择ETF", "请先在主要ETF表格中选择一行或多行。")
            return
        merged = parse_symbol_list([*parse_symbol_list(self.review_proxy_symbols.text()), *symbols])
        self.review_proxy_symbols.setText(", ".join(merged))
        self.review_summary.setText(f"已加入 ETF：{', '.join(symbols)}。")

    def _review_ai_config(self) -> LLMConfig:
        preset = _provider_preset_from_label(self.review_ai_provider.currentText())
        return LLMConfig(
            provider=preset.provider,
            api_key=self.review_ai_api_key.text().strip() or None,
            base_url=self.review_ai_base_url.text().strip(),
            model=self.review_ai_model.text().strip(),
            thinking=self.review_ai_thinking.isChecked(),
        )

    def _run_review_operation(self, request: ReviewRequest, ai_config: LLMConfig | None) -> dict[str, object]:
        bundle = self._service().analyze_review_bundle(request)
        ai_result = run_review_ai(bundle.reviews, ai_config) if ai_config is not None else None
        return {"bundle": bundle, "ai_result": ai_result}

    def _run_data_check(self) -> None:
        request = DataCoverageRequest(
            symbols=parse_symbol_list(self.data_symbols.toPlainText()),
            start=_date_text(self.data_start),
            end=_date_text(self.data_end),
        )
        self.data_coverage_summary.setText("检查中")
        self.data_coverage_detail_button.setEnabled(False)
        self._run_task(
            lambda: self._service().check_coverage(request),
            self._show_data_check_result,
            status_label=self.data_status,
            busy_text="正在检查覆盖...",
        )

    def _run_data_update(self) -> None:
        request = DataUpdateRequest(
            symbols=parse_symbol_list(self.data_symbols.toPlainText()),
            start=_date_text(self.data_start),
            end=_date_text(self.data_end),
            download_engine=self.data_download_engine.currentText(),
            provider=self.data_download_provider.text().strip(),
        )
        self.data_update_progress_bar.setRange(0, 0)
        self.data_update_progress_bar.setFormat("下载中")
        self._run_task(
            lambda: self._service().update_bars(request),
            self._show_data_update_result,
            status_label=self.data_status,
            busy_text="正在下载/更新行情...",
        )

    def _run_history_download(self) -> None:
        request = self._data_update_request(
            symbols=parse_symbol_list(self.history_symbol.toPlainText()),
            start=self.history_download_start.text().strip() or "2018-01-01",
            end=_date_text(self.history_as_of),
        )
        self._run_task(
            lambda: self._service().update_bars(request),
            self._show_history_download_result,
            status_label=self.history_status,
            busy_text="正在下载/更新历史行情...",
        )

    def _data_update_request(self, *, symbols: tuple[str, ...], start: str, end: str) -> DataUpdateRequest:
        return DataUpdateRequest(
            symbols=symbols,
            start=start,
            end=end,
            download_engine=self._current_download_engine(),
            provider=self._current_download_provider(),
        )

    def _current_download_engine(self) -> str:
        engine_input = getattr(self, "data_download_engine", None)
        return engine_input.currentText() if isinstance(engine_input, QComboBox) else "akshare"

    def _current_download_provider(self) -> str:
        engine = self._current_download_engine()
        provider_input = getattr(self, "data_download_provider", None)
        provider = provider_input.text().strip() if isinstance(provider_input, QLineEdit) else ""
        if engine == "tdx" and not provider:
            provider = self.tdx_path_input.text().strip()
        return provider

    def _run_price_import(self) -> None:
        request = PriceImportRequest(
            file_path=self.price_import_file.text().strip(),
            fallback_symbol=self.price_import_fallback_symbol.text().strip(),
        )
        self._run_task(
            lambda: self._service().import_price_file(request),
            self._show_price_import_result,
            status_label=self.price_import_status,
            busy_text="正在导入价格数据...",
        )

    def _full_daily_request(self) -> FullDailyUpdateRequest:
        return FullDailyUpdateRequest(
            start=_date_text(self.full_daily_start),
            end=_date_text(self.full_daily_end),
            download_engine="tdx",
            provider=self.full_daily_provider.text().strip() or self.tdx_path_input.text().strip(),
            batch_size=self.full_daily_batch_size.value(),
            skip_available=self.full_daily_skip_available.isChecked(),
            include_indexes=self.full_daily_include_indexes.isChecked(),
            extra_symbols=self.full_daily_extra_symbols.text().strip(),
        )

    def _plan_full_daily_update(self) -> None:
        request = self._full_daily_request()
        self.full_daily_summary.setText("预览中")
        self.full_daily_detail_button.setEnabled(False)
        self.full_daily_progress_bar.setRange(0, 100)
        self.full_daily_progress_bar.setValue(0)
        self.full_daily_progress_bar.setFormat("0/0")
        self._run_task(
            lambda: self._service().plan_full_daily_update(request),
            self._show_full_daily_plan,
            status_label=self.full_daily_status,
            busy_text="正在预览全A日线...",
        )

    def _run_full_daily_update(self) -> None:
        request = self._full_daily_request()
        self._begin_full_daily_pause_control()
        self.full_daily_progress_bar.setRange(0, 100)
        self.full_daily_progress_bar.setValue(0)
        self.full_daily_progress_bar.setFormat("0/0")
        self._run_task(
            lambda progress: self._service().run_full_daily_update(
                request,
                progress_callback=progress,
                pause_check=self._is_full_daily_paused,
            ),
            self._show_full_daily_update_result,
            status_label=self.full_daily_status,
            busy_text="正在下载全A日线...",
            on_progress=self._show_full_daily_progress,
            on_finished=self._end_full_daily_pause_control,
        )

    def _begin_full_daily_pause_control(self) -> None:
        self._full_daily_pause_event = threading.Event()
        self.full_daily_pause_button.setEnabled(True)
        self.full_daily_pause_button.setText("暂停下载")

    def _end_full_daily_pause_control(self) -> None:
        if self._full_daily_pause_event is not None:
            self._full_daily_pause_event.clear()
        self._full_daily_pause_event = None
        self.full_daily_pause_button.setEnabled(False)
        self.full_daily_pause_button.setText("暂停下载")

    def _toggle_full_daily_pause(self) -> None:
        if self._full_daily_pause_event is None:
            return
        if self._full_daily_pause_event.is_set():
            self._full_daily_pause_event.clear()
            self.full_daily_pause_button.setText("暂停下载")
            self.full_daily_status.setText("继续下载：当前批次完成后进入下一批。")
            return
        self._full_daily_pause_event.set()
        self.full_daily_pause_button.setText("继续下载")
        self.full_daily_status.setText("已暂停：当前批次结束后停止进入下一批。")

    def _is_full_daily_paused(self) -> bool:
        return bool(self._full_daily_pause_event is not None and self._full_daily_pause_event.is_set())

    def _preview_migration(self) -> None:
        self._run_task(
            lambda: self._service().plan_kline_migration(
                self.migration_source.text().strip(),
                self.migration_destination.text().strip(),
                overwrite=self.migration_overwrite.isChecked(),
            ),
            self._show_migration_result,
            status_label=self.migration_status,
            busy_text="正在生成迁移预览...",
        )

    def _execute_migration(self) -> None:
        self._run_task(
            lambda: self._service().migrate_kline_data(
                self.migration_source.text().strip(),
                self.migration_destination.text().strip(),
                mode=self.migration_mode.currentText(),
                overwrite=self.migration_overwrite.isChecked(),
            ),
            self._show_migration_result,
            status_label=self.migration_status,
            busy_text="正在执行迁移...",
        )

    def _run_task(
        self,
        operation: Callable[..., object],
        on_success: Callable[[Any], None],
        *,
        status_label: QLabel,
        busy_text: str = "正在运行...",
        on_progress: Callable[[Any], None] | None = None,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        status_label.setText(busy_text)
        worker = TaskWorker(operation, progress_enabled=on_progress is not None)
        self._workers.append(worker)
        worker.succeeded.connect(on_success)
        worker.failed.connect(lambda message: self._show_error(message, status_label))
        if on_progress is not None:
            worker.progressed.connect(on_progress)
        if on_finished is not None:
            worker.finished.connect(on_finished)
        worker.finished.connect(lambda: self._remove_worker(worker))
        worker.start()

    def _remove_worker(self, worker: TaskWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def _show_error(self, message: str, status_label: QLabel) -> None:
        status_label.setText(f"失败：{message}")
        QMessageBox.warning(self, "运行失败", message)

    def _show_history_result(self, result: object) -> None:
        raw_histories = result.get("histories") if isinstance(result, dict) else result
        chart_bars = result.get("bars") if isinstance(result, dict) else pd.DataFrame()
        coverage = result.get("coverage") if isinstance(result, dict) else pd.DataFrame()
        size_spread_bars = result.get("size_spread_bars") if isinstance(result, dict) else pd.DataFrame()
        size_spread_error = str(result.get("size_spread_error", "")) if isinstance(result, dict) else ""
        stock_names = result.get("stock_names", {}) if isinstance(result, dict) else {}
        histories = list(raw_histories) if isinstance(raw_histories, list) else [raw_histories]
        frames = []
        for history in histories:
            frame = history.results.copy() if hasattr(history, "results") else pd.DataFrame()
            if not frame.empty:
                target_symbol = str(getattr(history, "symbol", ""))
                frame.insert(0, "目标代码", target_symbol)
                frame.insert(1, "目标股票", _stock_name_from_map(stock_names, target_symbol))
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.history_table.setModel(DataFrameModel(frame))
        self.history_table.resizeColumnsToContents()
        first_history = histories[0] if histories else None
        self.history_chart.set_series(history_line_series(first_history) if first_history else [])
        self.history_kline_chart.set_series(
            history_candlestick_series(first_history, chart_bars, forward_bars=10, stock_names=stock_names) if first_history else []
        )
        spread = size_spread_frame(size_spread_bars if isinstance(size_spread_bars, pd.DataFrame) else pd.DataFrame())
        self.history_size_spread_chart.set_series(size_spread_line_series(spread))
        if first_history:
            stat_sections = dict(history_stat_frames(first_history))
            self.history_forward_stats_table.setModel(DataFrameModel(stat_sections.get("后验观察统计", pd.DataFrame())))
            self.history_bucket_stats_table.setModel(DataFrameModel(stat_sections.get("相似度分层表现", pd.DataFrame())))
            self.history_size_spread_table.setModel(
                DataFrameModel(size_spread_window_stats(spread, first_history.current_window, first_history.historical_windows))
            )
        self.history_data_table.setModel(DataFrameModel(data_status_frame(coverage if isinstance(coverage, pd.DataFrame) else pd.DataFrame())))
        for table in (self.history_forward_stats_table, self.history_bucket_stats_table, self.history_size_spread_table, self.history_data_table):
            table.resizeColumnsToContents()
        status = f"完成：{len(histories)} 个目标，命中 {len(frame)} 条。"
        self.history_status.setText(f"{status} {size_spread_error}".strip())

    def _show_cross_section_result(self, result: object) -> None:
        raw_cross_sections = result.get("cross_sections") if isinstance(result, dict) else result
        chart_bars = result.get("bars") if isinstance(result, dict) else pd.DataFrame()
        coverage = result.get("coverage") if isinstance(result, dict) else pd.DataFrame()
        stock_names = result.get("stock_names", {}) if isinstance(result, dict) else {}
        cross_sections = list(raw_cross_sections) if isinstance(raw_cross_sections, list) else [raw_cross_sections]
        frames = []
        for cross_section in cross_sections:
            frame = cross_section.results.copy() if hasattr(cross_section, "results") else pd.DataFrame()
            if not frame.empty:
                target_symbol = str(getattr(cross_section, "target_symbol", ""))
                frame.insert(0, "目标代码", target_symbol)
                frame.insert(1, "目标股票", _stock_name_from_map(stock_names, target_symbol))
                if "symbol" in frame.columns:
                    stock_name_values = frame["symbol"].map(lambda symbol: _stock_name_from_map(stock_names, symbol))
                    if "股票" in frame.columns:
                        frame["股票"] = stock_name_values
                    else:
                        insert_at = int(frame.columns.get_loc("symbol")) + 1
                        frame.insert(insert_at, "股票", stock_name_values)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.cross_table.setModel(DataFrameModel(frame))
        self.cross_table.resizeColumnsToContents()
        first_cross = cross_sections[0] if cross_sections else None
        self.cross_chart.set_bars(cross_section_score_bars(first_cross) if first_cross else [])
        self.cross_kline_chart.set_series(
            cross_section_candlestick_series(first_cross, chart_bars, max_matches=6, forward_bars=10, stock_names=stock_names)
            if first_cross
            else []
        )
        if first_cross:
            stat_sections = dict(cross_section_stat_frames(first_cross))
            self.cross_forward_stats_table.setModel(DataFrameModel(stat_sections.get("后验观察统计", pd.DataFrame())))
            self.cross_bucket_stats_table.setModel(DataFrameModel(stat_sections.get("相似度分层表现", pd.DataFrame())))
            self.cross_skipped_table.setModel(DataFrameModel(stat_sections.get("跳过样本", pd.DataFrame())))
        self.cross_data_table.setModel(DataFrameModel(data_status_frame(coverage if isinstance(coverage, pd.DataFrame) else pd.DataFrame())))
        self.cross_data_chart.set_bars(data_status_bars(coverage if isinstance(coverage, pd.DataFrame) else pd.DataFrame()))
        for table in (self.cross_forward_stats_table, self.cross_bucket_stats_table, self.cross_skipped_table, self.cross_data_table):
            table.resizeColumnsToContents()
        self.cross_status.setText(f"完成：{len(cross_sections)} 个目标，命中 {len(frame)} 条。")

    def _show_cross_coverage_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self.cross_data_table.setModel(DataFrameModel(data_status_frame(data)))
        self.cross_data_table.resizeColumnsToContents()
        self.cross_data_chart.set_bars(data_status_bars(data))
        status_counts = data.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not data.empty else {}
        summary = "，".join(f"{key} {value}" for key, value in status_counts.items()) or "无结果"
        self.cross_status.setText(f"覆盖检查完成：{summary}。")

    def _show_cross_download_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self.cross_data_table.setModel(DataFrameModel(data_status_frame(data)))
        self.cross_data_table.resizeColumnsToContents()
        self.cross_data_chart.set_bars(data_status_bars(data))
        status_counts = data.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not data.empty else {}
        summary = "，".join(f"{key} {value}" for key, value in status_counts.items()) or "无结果"
        self.cross_status.setText(f"下载完成：{summary}。")

    def _show_review_result(self, result: object) -> None:
        ai_result = result.get("ai_result") if isinstance(result, dict) else None
        bundle = result.get("bundle") if isinstance(result, dict) else None
        raw_reviews = getattr(bundle, "reviews", None) if bundle is not None else (result.get("reviews") if isinstance(result, dict) else result)
        reviews = list(raw_reviews) if isinstance(raw_reviews, list) else [raw_reviews]
        comparison_frame = getattr(bundle, "comparison_frame", pd.DataFrame()) if bundle is not None else pd.DataFrame()
        comparison_frames = getattr(bundle, "comparison_frames", []) if bundle is not None else []
        script_profiles = getattr(bundle, "script_profiles", []) if bundle is not None else []
        etf_matches = getattr(bundle, "etf_matches", pd.DataFrame()) if bundle is not None else pd.DataFrame()
        stock_names = getattr(bundle, "stock_names", {}) if bundle is not None else {}
        direction_by_symbol = getattr(bundle, "direction_by_symbol", {}) if bundle is not None else {}
        frame = review_segments_frame(reviews, stock_names=stock_names)
        self.review_table.setModel(DataFrameModel(frame))
        self.review_table.resizeColumnsToContents()
        review_kline_series = review_candlestick_series(reviews, stock_names=stock_names)
        self.review_chart.set_series(review_kline_series)
        self.review_relative_chart.set_series(review_relative_line_series(reviews, comparison_frames))
        self.review_overview_table.setModel(DataFrameModel(review_overview_frame(reviews, stock_names=stock_names)))
        self.review_ranking_table.setModel(
            DataFrameModel(review_ranking_frame(reviews, comparison_frame, stock_names=stock_names, direction_by_symbol=direction_by_symbol))
        )
        self.review_comparison_table.setModel(DataFrameModel(review_comparisons_frame(comparison_frame)))
        self.review_script_table.setModel(DataFrameModel(review_script_profile_frame(script_profiles)))
        self.review_auto_etf_table.setModel(DataFrameModel(etf_matches))
        for table in (
            self.review_overview_table,
            self.review_ranking_table,
            self.review_comparison_table,
            self.review_script_table,
            self.review_auto_etf_table,
        ):
            table.resizeColumnsToContents()
        if ai_result is not None:
            self.review_text.setMarkdown(_ai_result_markdown(ai_result))
            self.review_cards.set_cards(_ai_script_cards(ai_result, stock_names=stock_names), chart_series=review_kline_series)
        else:
            self.review_text.setMarkdown(
                "\n\n---\n\n".join(review_text(review, comparison_frame, stock_names=stock_names) for review in reviews)
            )
            self.review_cards.set_cards(
                review_critique_cards(
                    reviews,
                    comparison_frame,
                    stock_names=stock_names,
                    direction_by_symbol=direction_by_symbol,
                ),
                chart_series=review_kline_series,
            )
        valid_reviews = [review for review in reviews if not getattr(review, "window", pd.DataFrame()).empty]
        avg_return = pd.Series(
            [getattr(review, "overview", {}).get("return") for review in valid_reviews],
            dtype="float64",
        ).dropna()
        self.review_summary.setText(
            " | ".join(
                [
                    f"复盘 {len(reviews)} 个目标",
                    f"有效 {len(valid_reviews)} 个",
                    f"平均收益 {_format_percent(avg_return.mean()) if not avg_return.empty else '-'}",
                    f"波段 {len(frame)} 条",
                ]
            )
        )

    def _show_data_check_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self._data_coverage_detail_frame = data.copy()
        self.data_coverage_table.setModel(DataFrameModel(data_status_frame(data)))
        self.data_coverage_table.resizeColumnsToContents()
        self.data_coverage_chart.set_bars(data_status_bars(data))
        available = int((data.get("status", pd.Series(dtype=str)) == "available").sum()) if not data.empty else 0
        self.data_coverage_summary.setText(f"{available}/{len(data)} 可用")
        self.data_coverage_detail_button.setEnabled(not data.empty)
        self.data_status.setText(f"完成：检查 {len(data)} 个代码，完整覆盖 {available} 个。")

    def _show_data_update_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self._data_coverage_detail_frame = data.copy()
        self.data_coverage_table.setModel(DataFrameModel(data))
        self.data_coverage_table.resizeColumnsToContents()
        self.data_coverage_chart.set_bars(data_status_bars(data))
        status_counts = data.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not data.empty else {}
        summary = "，".join(f"{key} {value}" for key, value in status_counts.items()) or "无结果"
        success = int((data.get("status", pd.Series(dtype=str)).astype(str) == "success").sum()) if not data.empty else 0
        self.data_coverage_summary.setText(f"{success}/{len(data)} 下载成功")
        self.data_coverage_detail_button.setEnabled(not data.empty)
        self.data_update_progress_bar.setRange(0, 1)
        self.data_update_progress_bar.setValue(1)
        self.data_update_progress_bar.setFormat(f"{len(data)}/{len(data)}")
        self.data_status.setText(f"完成：{summary}。")

    def _show_history_download_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self.history_data_table.setModel(DataFrameModel(data_status_frame(data)))
        self.history_data_table.resizeColumnsToContents()
        status_counts = data.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not data.empty else {}
        summary = "，".join(f"{key} {value}" for key, value in status_counts.items()) or "无结果"
        self.history_status.setText(f"下载完成：{summary}。")

    def _show_price_import_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self.price_import_table.setModel(DataFrameModel(data))
        self.price_import_table.resizeColumnsToContents()
        imported = int((data.get("status", pd.Series(dtype=str)) == "imported").sum()) if not data.empty else 0
        self.price_import_status.setText(f"完成：导入 {imported} 个代码。")

    def _show_full_daily_plan(self, plan: object) -> None:
        coverage = getattr(plan, "coverage", pd.DataFrame())
        download_symbols = tuple(getattr(plan, "download_symbols", ()))
        if isinstance(coverage, pd.DataFrame) and not coverage.empty:
            frame = data_status_frame(coverage)
        else:
            frame = pd.DataFrame({"symbol": list(download_symbols), "status": ["pending"] * len(download_symbols)})
        self._full_daily_detail_frame = frame.copy()
        self.full_daily_table.setModel(DataFrameModel(frame))
        self.full_daily_table.resizeColumnsToContents()
        total_count = int(getattr(plan, "total_count", len(frame)))
        stock_count = int(getattr(plan, "stock_count", 0))
        index_count = int(getattr(plan, "index_count", 0))
        self.full_daily_summary.setText(f"{len(download_symbols)}/{total_count} 待下载")
        self.full_daily_detail_button.setEnabled(not frame.empty)
        self.full_daily_status.setText(
            f"预览完成：范围 {total_count} 个，待下载 {len(download_symbols)} 个，股票 {stock_count} 个，指数/额外 {index_count} 个。"
        )

    def _show_full_daily_update_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self._full_daily_detail_frame = data.copy()
        self.full_daily_table.setModel(DataFrameModel(data_status_frame(data)))
        self.full_daily_table.resizeColumnsToContents()
        status_counts = data.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not data.empty else {}
        summary = "，".join(f"{key} {value}" for key, value in status_counts.items()) or "无结果"
        available = int((data.get("status", pd.Series(dtype=str)).astype(str) == "available").sum()) if not data.empty else 0
        self.full_daily_summary.setText(f"{available}/{len(data)} 可用")
        self.full_daily_detail_button.setEnabled(not data.empty)
        self.full_daily_progress_bar.setRange(0, max(len(data), 1))
        self.full_daily_progress_bar.setValue(len(data))
        self.full_daily_progress_bar.setFormat(f"{len(data)}/{len(data)}")
        self.full_daily_status.setText(f"完成：{summary}。")

    def _show_full_daily_progress(self, progress: object) -> None:
        data = progress if isinstance(progress, dict) else {}
        completed = int(data.get("completed", 0) or 0)
        total = int(data.get("total", 0) or 0)
        batch_index = int(data.get("batch_index", 0) or 0)
        batch_count = int(data.get("batch_count", 0) or 0)
        current = str(data.get("current", "") or "").strip()
        prefix = "已暂停" if data.get("paused") is True else "全A下载进度"
        text = f"{prefix}：{completed}/{total}；批次 {batch_index}/{batch_count}"
        if current:
            text = f"{text}；当前 {current}"
        self.full_daily_progress_bar.setRange(0, max(total, 1))
        self.full_daily_progress_bar.setValue(min(completed, max(total, 1)))
        self.full_daily_progress_bar.setFormat(f"{completed}/{total}")
        self.full_daily_summary.setText(f"{completed}/{total} 已处理")
        self.full_daily_status.setText(text)

    def _show_migration_result(self, frame: object) -> None:
        data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        self.migration_table.setModel(DataFrameModel(data))
        self.migration_table.resizeColumnsToContents()
        status_counts = data.get("status", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if not data.empty else {}
        summary = "，".join(f"{key} {value}" for key, value in status_counts.items()) or "无文件"
        self.migration_status.setText(f"完成：{summary}。")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._has_running_workers():
            event.ignore()
            QMessageBox.information(self, "任务运行中", "后台任务还在运行，请等待完成后再关闭应用。")
            return
        super().closeEvent(event)

    def _has_running_workers(self) -> bool:
        return any(worker.isRunning() for worker in self._workers)


def main() -> int:
    multiprocessing.freeze_support()
    app = QApplication([])
    app.setWindowIcon(create_app_icon())
    window = MainWindow()
    window.setWindowIcon(app.windowIcon())
    window.show()
    return int(app.exec())


def _nav_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("navButton")
    button.setCheckable(False)
    return button


def _page_title(title: str, subtitle: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    title_label = QLabel(title)
    title_label.setObjectName("pageTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("pageSubtitle")
    layout.addWidget(title_label)
    layout.addWidget(subtitle_label)
    return widget


def _scroll_page(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setObjectName("pageScroll")
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def _chart_tab(widgets: tuple[QWidget, ...]) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return tab


def _text_output_panel(text: QTextEdit, toggle_button: QPushButton) -> QWidget:
    panel = QWidget()
    panel.setObjectName("resultTextPanel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    toolbar = QHBoxLayout()
    toolbar.setContentsMargins(0, 0, 0, 0)
    toolbar.addWidget(_field_label("复盘内容"))
    toolbar.addStretch(1)
    toolbar.addWidget(toggle_button)
    layout.addLayout(toolbar)
    layout.addWidget(text)
    return panel


def _toggle_panel_widget(widget: QWidget, button: QPushButton, collapsed_text: str, expanded_text: str) -> None:
    next_collapsed = not bool(widget.property("collapsed"))
    widget.setProperty("collapsed", next_collapsed)
    widget.setVisible(not next_collapsed)
    button.setText(collapsed_text if next_collapsed else expanded_text)


def _table_tab(table: QTableView, expand_button: QPushButton) -> QWidget:
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    toolbar = QHBoxLayout()
    toolbar.addStretch(1)
    toolbar.addWidget(expand_button)
    layout.addLayout(toolbar)
    layout.addWidget(table, stretch=1)
    return tab


def _multi_table_tab(sections: tuple[tuple[str, QTableView], ...]) -> QWidget:
    tab = QWidget()
    tab.setObjectName("statsTab")
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    inner_tabs = QTabWidget()
    inner_tabs.setObjectName("statsWorkbench")
    for title, table in sections:
        section = QFrame()
        section.setObjectName("tablePanel")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(12, 10, 12, 12)
        section_layout.setSpacing(8)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("tablePanelTitle")
        meta_label = QLabel("横向滚动 / 表头排序 / 展开查看全表")
        meta_label.setObjectName("tablePanelMeta")
        toolbar.addWidget(title_label)
        toolbar.addWidget(meta_label)
        toolbar.addStretch(1)
        expand_button = QPushButton("展开统计")
        expand_button.setObjectName("flatActionButton")
        expand_button.clicked.connect(lambda checked=False, source=table, name=title: _open_table_dialog_window(source, name))
        toolbar.addWidget(expand_button)
        section_layout.addLayout(toolbar)
        section_layout.addWidget(table, stretch=1)
        inner_tabs.addTab(section, title)
    layout.addWidget(inner_tabs, stretch=1)
    return tab


def _open_table_dialog_window(table: QTableView, title: str) -> None:
    model = table.model()
    if not isinstance(model, DataFrameModel) or model.frame().empty:
        QMessageBox.information(table.window(), "没有可展开数据", "当前表格为空。")
        return
    dialog = _dataframe_detail_dialog(model.frame(), title, table.window())
    dialog.exec()


def _dataframe_detail_dialog(frame: pd.DataFrame, title: str, parent: QWidget | None) -> QDialog:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(1240, 760)
    source_model = DataFrameModel(frame)
    proxy = DataFrameFilterProxyModel()
    source_model.setParent(dialog)
    proxy.setParent(dialog)
    proxy.setSourceModel(source_model)

    search = QLineEdit()
    search.setObjectName("detailSearchInput")
    search.setPlaceholderText("查找代码、名称、状态或提示")
    search.textChanged.connect(proxy.set_search_text)

    status_filter = QComboBox()
    status_filter.setObjectName("detailStatusFilter")
    statuses = sorted(str(value) for value in frame.get("status", pd.Series(dtype=str)).dropna().unique())
    status_filter.addItems(["全部", *statuses])
    status_filter.currentTextChanged.connect(proxy.set_status_filter)

    table = _table()
    table.setObjectName("detailTable")
    table.setModel(proxy)
    table.resizeColumnsToContents()

    toolbar = QHBoxLayout()
    toolbar.addWidget(search, stretch=1)
    toolbar.addWidget(status_filter)
    layout = QVBoxLayout(dialog)
    layout.addLayout(toolbar)
    layout.addWidget(table, stretch=1)
    return dialog


def _kline_toolbar(combo: QComboBox, chart: CandlestickChartWidget) -> QWidget:
    toolbar = QWidget()
    layout = QHBoxLayout(toolbar)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(_field_label("K线布局"))
    layout.addWidget(combo)
    for label in ("1*2", "2*2", "2*4", "3*3"):
        button = QPushButton(label)
        button.setObjectName("klineLayoutShortcut")
        button.clicked.connect(lambda checked=False, value=label: combo.setCurrentText(value))
        layout.addWidget(button)
    expand_button = QPushButton("展开K线")
    expand_button.setObjectName("klineExpandButton")

    def toggle_chart() -> None:
        chart.set_expanded(not chart.is_expanded())
        expand_button.setText("收起K线" if chart.is_expanded() else "展开K线")

    expand_button.clicked.connect(toggle_chart)
    layout.addWidget(expand_button)
    layout.addStretch(1)
    return toolbar


def _metric_card(title: str, value: QLabel, caption: str) -> QWidget:
    card = QFrame()
    card.setObjectName("metricCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(8)
    title_label = QLabel(title)
    title_label.setObjectName("metricTitle")
    value.setObjectName("metricValue")
    caption_label = QLabel(caption)
    caption_label.setObjectName("metricCaption")
    caption_label.setWordWrap(True)
    layout.addWidget(title_label)
    layout.addWidget(value)
    layout.addWidget(caption_label)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return card


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


def _module_card(module: OverviewModule, on_open: Callable[[int], None]) -> QWidget:
    card = QFrame()
    card.setObjectName("moduleCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)

    chart_badge = QLabel(module.chart_title)
    chart_badge.setObjectName("moduleBadge")
    title = QLabel(module.title)
    title.setObjectName("moduleTitle")
    summary = QLabel(module.summary)
    summary.setObjectName("moduleSummary")
    summary.setWordWrap(True)
    action = QPushButton(module.action_label)
    action.setObjectName("flatActionButton")
    action.clicked.connect(lambda checked=False, page_index=module.page_index: on_open(page_index))

    layout.addWidget(chart_badge, alignment=Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(title)
    layout.addWidget(summary)
    layout.addStretch(1)
    layout.addWidget(action, alignment=Qt.AlignmentFlag.AlignLeft)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return card


def _data_operation_panel(title: str, subtitle: str, form_layout: QGridLayout) -> QWidget:
    panel = QFrame()
    panel.setObjectName("dataOperationPanel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(10)

    title_label = QLabel(title)
    title_label.setObjectName("dataPanelTitle")
    subtitle_label = QLabel(subtitle)
    subtitle_label.setObjectName("dataPanelSubtitle")
    subtitle_label.setWordWrap(True)

    layout.addWidget(title_label)
    layout.addWidget(subtitle_label)
    layout.addLayout(form_layout)
    return panel


def _kline_layout_combo(chart: CandlestickChartWidget) -> QComboBox:
    combo = QComboBox()
    combo.setObjectName("klineLayoutCombo")
    combo.addItems(["1*2", "2*2", "2*4", "3*3"])
    combo.setEditable(True)
    combo.setMinimumWidth(96)
    if combo.lineEdit() is not None:
        combo.lineEdit().setTextMargins(8, 0, 8, 0)
    combo.setCurrentText("2*2")
    combo.currentTextChanged.connect(lambda label: chart.set_grid(*parse_kline_grid(label)))
    chart.set_grid(*parse_kline_grid(combo.currentText()))
    return combo


def parse_kline_grid(label: str) -> tuple[int, int]:
    left, _, right = str(label).replace("×", "*").partition("*")
    rows = int(left.strip() or "2")
    columns = int(right.strip() or left.strip() or "2")
    return max(1, rows), max(1, columns)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not items or any(item <= 0 for item in items):
        raise ValueError("参数必须是逗号分隔的正整数。")
    return items


def _benchmark_command(data_root: str, timeframe: str, adjust: str) -> str:
    return (
        "python -m ashare_cross_section_similarity benchmark "
        "--cases docs/research/similarity_benchmark_cases.yaml "
        "--algorithms baseline_price_feature,return_shape,hybrid_shape_v2,dtw_optional "
        f"--data-root {data_root} --timeframe {timeframe} --adjust {adjust} "
        "--output outputs/research"
    )


@dataclass(frozen=True)
class _ReviewCardPalette:
    accent: str
    border: str
    background: str
    badge_background: str
    badge_text: str


def _review_card_palette(grade: str) -> _ReviewCardPalette:
    normalized = str(grade or "").strip().upper()
    grade_classes = {
        "夯爆了": "grade-s",
        "S": "grade-s",
        "人上人": "grade-a",
        "A": "grade-a",
        "立棍单打": "grade-b",
        "B": "grade-b",
        "刷子": "grade-c",
        "C": "grade-c",
        "路边": "grade-d",
        "D": "grade-d",
        "混子": "grade-d",
        "NPC": "grade-e",
        "E": "grade-e",
        "拉完了": "grade-f",
        "F": "grade-f",
    }
    palette = {
        "grade-s": _ReviewCardPalette("#7c3aed", "#ddd6fe", "#f5f3ff", "#ede9fe", "#5b21b6"),
        "grade-a": _ReviewCardPalette("#16a34a", "#bbf7d0", "#f0fdf4", "#dcfce7", "#166534"),
        "grade-b": _ReviewCardPalette("#2563eb", "#bfdbfe", "#eff6ff", "#dbeafe", "#1d4ed8"),
        "grade-c": _ReviewCardPalette("#d97706", "#fde68a", "#fffbeb", "#fef3c7", "#92400e"),
        "grade-d": _ReviewCardPalette("#64748b", "#cbd5e1", "#f8fafc", "#e2e8f0", "#334155"),
        "grade-e": _ReviewCardPalette("#475569", "#cbd5e1", "#f8fafc", "#e2e8f0", "#1e293b"),
        "grade-f": _ReviewCardPalette("#dc2626", "#fecaca", "#fef2f2", "#fee2e2", "#991b1b"),
    }
    grade_class = grade_classes.get(str(grade or "").strip(), grade_classes.get(normalized, "grade-d"))
    return palette[grade_class]


def _review_card_widget(card: ReviewCritiqueCard, on_expand: Callable[[], None] | None = None) -> QWidget:
    frame = QFrame()
    frame.setObjectName("reviewCritiqueCard")
    palette = _review_card_palette(card.grade)
    frame.setStyleSheet(
        f"""
QFrame#reviewCritiqueCard {{
    background: {palette.background};
    border: 1px solid {palette.border};
    border-left: 5px solid {palette.accent};
    border-radius: 8px;
}}
"""
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)

    head = QHBoxLayout()
    title = QLabel(card.title)
    title.setObjectName("reviewCardTitle")
    title.setWordWrap(True)
    grade = QLabel(card.grade or "-")
    grade.setObjectName("reviewCardGrade")
    grade.setStyleSheet(
        f"""
QLabel#reviewCardGrade {{
    background: {palette.badge_background};
    color: {palette.badge_text};
    border-radius: 10px;
    padding: 4px 8px;
    font-weight: 800;
}}
"""
    )
    head.addWidget(title, stretch=1)
    head.addWidget(grade)
    layout.addLayout(head)

    nature = QLabel(card.nature or "-")
    nature.setObjectName("reviewCardNature")
    nature.setWordWrap(True)
    layout.addWidget(nature)

    critique = QLabel(card.critique or "-")
    critique.setObjectName("reviewCardCritique")
    critique.setWordWrap(True)
    layout.addWidget(critique)

    metrics = QGridLayout()
    metrics.setHorizontalSpacing(6)
    metrics.setVerticalSpacing(6)
    for index, (label, value) in enumerate(card.metrics):
        metric = QLabel(f"{label}\n{value}")
        metric.setObjectName("reviewCardMetric")
        metrics.addWidget(metric, index // 2, index % 2)
    layout.addLayout(metrics)
    if on_expand is not None:
        expand = _form_button("全屏K线", maximum_width=116)
        expand.setObjectName("reviewCardExpandButton")
        expand.clicked.connect(lambda checked=False: on_expand())
        layout.addWidget(expand, alignment=Qt.AlignmentFlag.AlignLeft)
    return frame


def _review_kline_dialog(card: ReviewCritiqueCard, chart_series: list[CandlestickSeries], parent: QWidget) -> QDialog:
    dialog = QDialog(parent)
    dialog.setObjectName("reviewKlineDialog")
    dialog.setWindowTitle(f"K线复盘 - {card.title or card.symbol}")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)

    header = QHBoxLayout()
    title = QLabel(card.title or card.symbol)
    title.setObjectName("reviewDialogTitle")
    export = _form_button("导出 PNG", maximum_width=116)
    export.setObjectName("reviewCardExportPngButton")
    header.addWidget(title, stretch=1)
    header.addWidget(export)
    layout.addLayout(header)

    chart = CandlestickChartWidget("K线复盘")
    chart.setObjectName("reviewKlineDialogChart")
    chart.set_expanded(True)
    chart.set_grid(1, 1)
    chart.set_series(chart_series)
    export.clicked.connect(lambda checked=False, source=chart: _export_widget_png(source))
    layout.addWidget(chart, stretch=1)
    return dialog


def _export_widget_png(widget: QWidget) -> None:
    path, _ = QFileDialog.getSaveFileName(
        widget,
        "导出 PNG",
        "review_card.png",
        "PNG 图片 (*.png);;所有文件 (*)",
    )
    if not path:
        return
    if not str(path).lower().endswith(".png"):
        path = f"{path}.png"
    if not _save_widget_png(widget, path):
        QMessageBox.warning(widget, "导出失败", "PNG 截图保存失败。")


def _save_widget_png(widget: QWidget, path: str | Path) -> bool:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(widget.grab().save(str(output), "PNG"))


def _series_by_symbol(series: list[CandlestickSeries]) -> dict[str, list[CandlestickSeries]]:
    result: dict[str, list[CandlestickSeries]] = {}
    for item in series:
        symbol = _symbol_from_card_title(item.label)
        if symbol:
            result.setdefault(symbol, []).append(item)
    return result


def _clear_layout(layout: QGridLayout | QVBoxLayout | QHBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


def _provider_preset_from_label(label: str):
    for preset in provider_presets().values():
        if preset.label == label:
            return preset
    return provider_presets()[DEFAULT_LLM_PROVIDER]


def _ai_result_markdown(result: object) -> str:
    sections = [
        ("AI 复盘", getattr(result, "review", "")),
        ("AI 分析", getattr(result, "analysis", "")),
        ("AI 锐评", getattr(result, "critique", "")),
    ]
    lines: list[str] = []
    for title, body in sections:
        if str(body).strip():
            lines.append(f"## {title}\n\n{body}")
    cards = getattr(result, "script_cards", ())
    if cards:
        lines.append("## 视频脚本卡")
        for card in cards:
            grade = getattr(card, "grade", "")
            title = getattr(card, "title", "")
            body = getattr(card, "body", "")
            lines.append(f"**{title}**（{grade}）\n\n{body}")
    disclaimer = getattr(result, "disclaimer", "")
    if str(disclaimer).strip():
        lines.append(f"_{disclaimer}_")
    return "\n\n".join(lines)


def _ai_script_cards(result: object, *, stock_names: dict[str, str] | None = None) -> list[ReviewCritiqueCard]:
    cards = getattr(result, "script_cards", ()) or ()
    output: list[ReviewCritiqueCard] = []
    for card in cards:
        raw_title = str(getattr(card, "title", "") or "").strip()
        symbol = _symbol_from_card_title(raw_title)
        name = _stock_name_from_map(stock_names or {}, symbol)
        title = f"{name}（{symbol}）" if symbol and name else raw_title
        grade = str(getattr(card, "grade", "") or "").strip()
        body = str(getattr(card, "body", "") or "").strip()
        output.append(
            ReviewCritiqueCard(
                symbol=symbol or raw_title,
                title=title,
                grade=grade,
                nature="AI 锐评",
                critique=body,
                metrics=(("档位", grade or "-"), ("来源", "AI"), ("用途", "脚本"), ("校验", "证据链")),
            )
        )
    return output


def _symbol_from_card_title(title: str) -> str:
    match = re.search(r"(?P<code>\d{6})(?:[._](?P<exchange>SH|SZ|BJ))?", str(title or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    code = match.group("code")
    exchange = match.group("exchange")
    return _normalize_card_symbol(f"{code}.{exchange}") if exchange else _normalize_card_symbol(code)


def _normalize_card_symbol(value: object) -> str:
    symbols = parse_symbol_list((value,))
    return symbols[0] if symbols else ""


def _field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


def _form_button(text: str, *, primary: bool = False, maximum_width: int = 160) -> QPushButton:
    button = QPushButton(text)
    if primary:
        button.setObjectName("primaryButton")
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setMaximumWidth(maximum_width)
    button.setMinimumWidth(min(96, maximum_width))
    return button


def _progress_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setObjectName("cardProgressBar")
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setFormat("0/0")
    bar.setTextVisible(True)
    bar.setMinimumHeight(30)
    return bar


def _symbol_text(value: str) -> QPlainTextEdit:
    edit = QPlainTextEdit(value)
    edit.setFixedHeight(68)
    edit.setPlaceholderText("多个代码用半角逗号分隔，如 300750.SZ, 600519.SH")
    return edit


def _add_form_row(layout: QGridLayout, row: int, left_label: str, left: QWidget, right_label: str, right: QWidget) -> None:
    layout.addWidget(_field_label(left_label), row, 0)
    layout.addWidget(left, row, 1)
    layout.addWidget(_field_label(right_label), row, 2)
    layout.addWidget(right, row, 3)
    layout.setColumnStretch(1, 1)
    layout.setColumnStretch(3, 1)


def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
    return spin


def _double_spin(minimum: float, maximum: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setSingleStep(step)
    spin.setDecimals(2)
    spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
    return spin


def _algorithm_combo(*, mode: str) -> QComboBox:
    combo = QComboBox()
    choices = ALGORITHM_CHOICES if mode == "history" else tuple(item for item in ALGORITHM_CHOICES if item != "mass_optional_history")
    for name in choices:
        combo.addItem(algorithm_label(name), name)
    default_index = max(0, combo.findData(BASELINE_ALGORITHM))
    combo.setCurrentIndex(default_index)
    return combo


def _date_edit(value: QDate) -> DatePicker:
    return DatePicker(value)


def _configure_calendar(calendar: QCalendarWidget) -> None:
    calendar.setGridVisible(True)
    calendar.setMinimumDate(QDate(1990, 1, 1))
    calendar.setMaximumDate(QDate.currentDate().addYears(2))
    calendar.setMinimumWidth(360)
    calendar.setMinimumHeight(320)


def _date_text(edit: DatePicker) -> str:
    return edit.date_text()


def _today_text() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _stock_name_from_map(stock_names: object, symbol: object) -> str:
    names = stock_names if isinstance(stock_names, dict) else {}
    raw_symbol = str(symbol or "").strip()
    normalized = parse_symbol_list((raw_symbol,))
    key = normalized[0] if normalized else raw_symbol
    return str(names.get(key) or names.get(raw_symbol) or "").strip()


def _table() -> QTableView:
    table = QTableView()
    table.setObjectName("resultTable")
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(True)
    table.setMinimumHeight(380)
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideNone)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setMinimumSectionSize(92)
    table.horizontalHeader().setStretchLastSection(False)
    table.verticalHeader().setDefaultSectionSize(32)
    return table


def _compact_table() -> QTableView:
    table = _table()
    table.setMinimumHeight(180)
    return table


def _draw_panel(painter: QPainter, rect: QRectF) -> None:
    painter.setPen(QPen(QColor("#e1e4dd"), 1))
    painter.setBrush(QColor("#ffffff"))
    painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 8, 8)


def _draw_chart_title(painter: QPainter, title: str, rect: QRectF) -> None:
    painter.setPen(QColor("#25322e"))
    font = painter.font()
    font.setBold(True)
    font.setPointSize(11)
    painter.setFont(font)
    painter.drawText(rect.adjusted(16, 14, -16, -8), Qt.AlignmentFlag.AlignLeft, title)


def _draw_grid(painter: QPainter, plot: QRectF) -> None:
    painter.setPen(QPen(QColor("#eef1eb"), 1))
    for step in range(1, 4):
        y = plot.top() + plot.height() * step / 4
        painter.drawLine(plot.left(), int(y), plot.right(), int(y))


def _draw_empty_chart(painter: QPainter, plot: QRectF, text: str) -> None:
    painter.setPen(QColor("#87918b"))
    painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, text)


def _draw_candlestick_panel(
    painter: QPainter,
    panel: QRectF,
    series: CandlestickSeries,
    visible_left: float,
    visible_right: float,
) -> None:
    painter.setPen(QPen(QColor("#eef1eb"), 1))
    painter.setBrush(QColor("#fbfcfa"))
    painter.drawRoundedRect(panel, 6, 6)
    painter.setPen(QColor("#25322e"))
    painter.drawText(panel.adjusted(10, 9, -10, -8), Qt.AlignmentFlag.AlignLeft, series.label)

    plot = panel.adjusted(12, 32, -12, -20)
    rows = _visible_candles(series, visible_left, visible_right)
    if not rows:
        _draw_empty_chart(painter, plot, "暂无K线数据")
        return
    lows = [row.low for row in rows]
    highs = [row.high for row in rows]
    min_value = min(lows)
    max_value = max(highs)
    if min_value == max_value:
        min_value -= 1
        max_value += 1
    candle_step = plot.width() / max(len(rows), 1)
    candle_width = max(3, min(12, candle_step * 0.62))
    _draw_candlestick_highlights(painter, plot, rows, candle_step, series.highlights)
    _draw_grid(painter, plot)
    span = max_value - min_value
    for index, row in enumerate(rows):
        x = plot.left() + candle_step * index + candle_step / 2
        high_y = plot.bottom() - plot.height() * (row.high - min_value) / span
        low_y = plot.bottom() - plot.height() * (row.low - min_value) / span
        open_y = plot.bottom() - plot.height() * (row.open - min_value) / span
        close_y = plot.bottom() - plot.height() * (row.close - min_value) / span
        color = QColor("#d64545") if row.close >= row.open else QColor("#2d6a4f")
        painter.setPen(QPen(color, 1.2))
        painter.drawLine(int(x), int(high_y), int(x), int(low_y))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        body_top = min(open_y, close_y)
        body_height = max(2.0, abs(close_y - open_y))
        painter.drawRoundedRect(QRectF(x - candle_width / 2, body_top, candle_width, body_height), 1.5, 1.5)
    painter.setPen(QColor("#87918b"))
    start_date = rows[0].date.strftime("%Y-%m-%d")
    end_date = rows[-1].date.strftime("%Y-%m-%d")
    painter.drawText(plot.adjusted(0, plot.height() + 3, 0, 18), Qt.AlignmentFlag.AlignLeft, start_date)
    painter.drawText(plot.adjusted(0, plot.height() + 3, 0, 18), Qt.AlignmentFlag.AlignRight, end_date)


def _draw_candlestick_highlights(
    painter: QPainter,
    plot: QRectF,
    rows: list,
    candle_step: float,
    highlights: tuple,
) -> None:
    if not rows or not highlights:
        return
    dates = [row.date.normalize() for row in rows]
    for highlight in highlights:
        start = pd.Timestamp(highlight.start).normalize()
        end = pd.Timestamp(highlight.end).normalize()
        indexes = [index for index, date in enumerate(dates) if start <= date <= end]
        if not indexes:
            continue
        color = QColor(str(highlight.color))
        color.setAlphaF(_clamp(float(highlight.opacity), 0.05, 0.6))
        left = plot.left() + candle_step * min(indexes)
        right = plot.left() + candle_step * (max(indexes) + 1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRect(QRectF(left, plot.top(), max(candle_step, right - left), plot.height()))


def _visible_candles(series: CandlestickSeries, visible_left: float, visible_right: float) -> list:
    rows = list(series.rows)
    if not rows:
        return []
    start = int(len(rows) * _clamp(visible_left, 0.0, 1.0))
    end = max(start + 1, int(len(rows) * _clamp(visible_right, 0.0, 1.0) + 0.999999))
    return rows[start:end]


def _line_points(values: tuple[float, ...], plot: QRectF, min_value: float, max_value: float) -> list[tuple[int, int]]:
    if len(values) == 1:
        return [(plot.left() + plot.width() // 2, plot.center().y())]
    points: list[tuple[int, int]] = []
    span = max_value - min_value
    for index, value in enumerate(values):
        x = plot.left() + plot.width() * index / (len(values) - 1)
        y = plot.bottom() - plot.height() * (value - min_value) / span
        points.append((int(x), int(y)))
    return points


def _draw_legend(painter: QPainter, series: list[LineSeries], palette: list[QColor], rect: QRectF) -> None:
    x = rect.left() + 100
    y = rect.top() + 25
    painter.setFont(painter.font())
    for index, item in enumerate(series[:4]):
        color = palette[index % len(palette)]
        painter.setPen(QPen(color, 2))
        painter.drawLine(int(x), int(y - 4), int(x + 16), int(y - 4))
        painter.setPen(QColor("#52605a"))
        painter.drawText(int(x + 22), int(y), item.label)
        x += 88


def _chart_palette() -> list[QColor]:
    return [QColor("#2d6a4f"), QColor("#6b7f3f"), QColor("#2563eb"), QColor("#b45309")]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), minimum), maximum)


def _format_cell(value: object, column: str) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        if _is_percent_column(column):
            return _format_percent(value)
        if abs(value) < 1:
            return f"{value:.4f}"
        return f"{value:.2f}"
    return str(value)


def _is_percent_column(column: str) -> bool:
    if "相关" in column:
        return False
    return any(key in column for key in ("收益", "回撤", "波动", "覆盖", "占比", "胜率", "浮盈", "相似度", "超额", "return", "drawdown", "volatility"))


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not pd.notna(number):
        return ""
    return f"{number:.2%}"


def _style_sheet() -> str:
    return """
QMainWindow, QWidget#content {
    background: #f6f7f4;
    color: #1f2933;
    font-family: "Inter", "SF Pro Display", "PingFang SC", "Microsoft YaHei";
    font-size: 14px;
}
QScrollArea#pageScroll {
    background: transparent;
    border: none;
}
QScrollArea#pageScroll > QWidget > QWidget {
    background: transparent;
}
QFrame#sidebar {
    background: #18201d;
    border: none;
}
QLabel#brandTitle {
    color: #f5f7f2;
    font-size: 22px;
    font-weight: 800;
}
QLabel#brandSubtitle {
    color: #aeb9b1;
    font-size: 13px;
}
QPushButton#navButton {
    background: transparent;
    color: #dce4de;
    border: none;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    font-weight: 650;
}
QPushButton#navButton:hover, QPushButton#navButton[active="true"] {
    background: #25322e;
    color: #ffffff;
}
QLabel#apiHint {
    color: #c8d2cc;
    background: #222c28;
    border: 1px solid #35453f;
    border-radius: 8px;
    padding: 12px;
    line-height: 1.4;
}
QFrame#header, QFrame#metricCard {
    background: #ffffff;
    border: 1px solid #e1e4dd;
    border-radius: 8px;
}
QFrame#moduleCard {
    background: #ffffff;
    border: 1px solid #e1e4dd;
    border-radius: 8px;
}
QFrame#moduleCard:hover {
    border-color: #b9c8bd;
    background: #fbfcfa;
}
QFrame#dataOperationPanel {
    background: #ffffff;
    border: 1px solid #d8ded6;
    border-left: 4px solid #2d6a4f;
    border-radius: 4px;
}
QLabel#pageTitle {
    color: #17201b;
    font-size: 24px;
    font-weight: 800;
}
QLabel#pageSubtitle, QLabel#mutedText, QLabel#metricCaption, QLabel#moduleSummary {
    color: #66706a;
}
QLabel#dataPanelTitle {
    color: #17201b;
    font-size: 16px;
    font-weight: 850;
}
QLabel#dataPanelSubtitle {
    color: #66706a;
    font-size: 12px;
    font-weight: 650;
}
QLabel#dataRootHint {
    background: #f7faf7;
    border: 1px solid #d8ded6;
    border-left: 4px solid #6b7f3f;
    border-radius: 4px;
    color: #40514a;
    padding: 9px 11px;
    font-size: 12px;
    font-weight: 650;
}
QLabel#coverageSummary {
    color: #17201b;
    font-size: 20px;
    font-weight: 850;
}
QLabel#fieldLabel, QLabel#metricTitle, QLabel#sectionLabel {
    color: #4d5852;
    font-size: 12px;
    font-weight: 700;
}
QLabel#sectionLabel {
    padding-top: 6px;
}
QLabel#metricValue {
    color: #17201b;
    font-size: 26px;
    font-weight: 800;
}
QLabel#moduleTitle {
    color: #17201b;
    font-size: 18px;
    font-weight: 800;
}
QLabel#moduleBadge {
    background: #eef3ed;
    color: #2d6a4f;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
    font-weight: 750;
}
QLabel#summaryText {
    background: #ffffff;
    border: 1px solid #e1e4dd;
    border-radius: 8px;
    padding: 12px;
    color: #23312b;
    font-weight: 650;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background: #ffffff;
    border: 1px solid #d5dad2;
    border-radius: 6px;
    padding: 9px 10px;
    color: #17241e;
    selection-background-color: #2d6a4f;
    selection-color: #ffffff;
}
QComboBox {
    background: transparent;
    border: none;
    border-bottom: 1px solid #d5dad2;
    border-radius: 0;
    padding: 9px 18px 9px 2px;
    color: #21342b;
    font-weight: 650;
    selection-background-color: #dfe8e1;
}
QComboBox:hover {
    background: transparent;
    border-bottom-color: #9fb2a6;
}
QComboBox:focus {
    background: transparent;
    border-bottom: 1px solid #2d6a4f;
}
QComboBox::drop-down {
    border: none;
    width: 0px;
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #dfe4dc;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #e1ebe3;
    selection-color: #17241e;
    outline: 0;
}
QComboBox QLineEdit {
    background: transparent;
    border: none;
    padding: 0;
    color: #21342b;
    selection-background-color: #dfe8e1;
    selection-color: #17241e;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border-color: #2d6a4f;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    color: #7d8782;
    background: #f4f6f2;
}
QPushButton#datePickButton {
    padding: 9px 12px;
}
QCalendarWidget {
    background: #ffffff;
    color: #17241e;
}
QCalendarWidget QToolButton {
    min-height: 30px;
    padding: 4px 8px;
    color: #17241e;
}
QCalendarWidget QSpinBox {
    min-height: 30px;
    padding: 4px 8px;
    color: #17241e;
}
QTabWidget::pane {
    border: 1px solid #e1e4dd;
    border-radius: 8px;
    background: #ffffff;
    top: -1px;
}
QWidget#statsTab {
    background: #f8fafc;
}
QTabWidget#statsWorkbench::pane {
    border: 1px solid #cbd5e1;
    border-radius: 0;
    background: #f8fafc;
}
QTabWidget#statsWorkbench QTabBar::tab {
    background: #e5e7eb;
    color: #334155;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    border-radius: 0;
    padding: 7px 14px;
    margin-right: 0;
    font-weight: 750;
}
QTabWidget#statsWorkbench QTabBar::tab:selected {
    background: #f8fafc;
    color: #0f172a;
}
QTabBar::tab {
    background: #eef3ed;
    color: #33413b;
    border: 1px solid #dfe4dc;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 9px 18px;
    margin-right: 4px;
    font-weight: 700;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #17241e;
}
QScrollBar:vertical {
    background: #eef3ed;
    width: 14px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #b8c8be;
    border-radius: 7px;
    min-height: 36px;
}
QScrollBar::handle:vertical:hover {
    background: #8fa596;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #eef3ed;
    height: 14px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #b8c8be;
    border-radius: 7px;
    min-width: 36px;
}
QScrollBar::handle:horizontal:hover {
    background: #8fa596;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QPushButton {
    background: #eef3ed;
    border: none;
    border-radius: 6px;
    padding: 10px 15px;
    color: #21342b;
    font-weight: 650;
}
QPushButton:hover {
    background: #e1ebe3;
    color: #17241e;
}
QPushButton:pressed {
    background: #d5e2d8;
}
QPushButton#primaryButton {
    background: #2d6a4f;
    color: #ffffff;
    border: none;
}
QPushButton#primaryButton:hover {
    background: #23543f;
}
QPushButton#primaryButton:pressed {
    background: #1d4534;
}
QProgressBar#cardProgressBar {
    background: #eef3ed;
    border: 1px solid #d5dad2;
    border-radius: 4px;
    color: #21342b;
    font-weight: 750;
    text-align: center;
}
QProgressBar#cardProgressBar::chunk {
    background: #2d6a4f;
    border-radius: 4px;
}
QPushButton#flatActionButton {
    background: transparent;
    color: #2d6a4f;
    border: 1px solid #cbd8cf;
    padding: 8px 12px;
}
QPushButton#flatActionButton:hover {
    background: #eef3ed;
    border-color: #b9c8bd;
}
QPushButton#klineLayoutShortcut, QPushButton#klineExpandButton {
    background: transparent;
    border: 1px solid #cbd8cf;
    border-radius: 4px;
    padding: 7px 10px;
    color: #2d4f3d;
}
QPushButton#klineLayoutShortcut:hover, QPushButton#klineExpandButton:hover {
    background: #eef3ed;
}
QFrame#tablePanel {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 0;
}
QLabel#tablePanelTitle {
    color: #0f172a;
    font-size: 13px;
    font-weight: 850;
}
QLabel#tablePanelMeta {
    color: #64748b;
    font-size: 12px;
    font-weight: 650;
    padding-left: 12px;
}
QTableView#resultTable, QTableView#reviewEtfPopupTable, QTableView#detailTable {
    background: #fbfdff;
    alternate-background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 0;
    gridline-color: #d7dee8;
    color: #0f172a;
    font-family: "JetBrains Mono", "SF Mono", "Menlo", "Consolas", "PingFang SC", monospace;
    font-size: 12px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QWidget#chartWidget {
    background: #ffffff;
    border: none;
}
QWidget#reviewCards {
    background: transparent;
}
QFrame#reviewCritiqueCard {
    background: #ffffff;
    border: 1px solid #dde4dc;
    border-left: 5px solid #2d6a4f;
    border-radius: 8px;
}
QLabel#reviewCardTitle {
    color: #17201b;
    font-weight: 800;
    font-size: 15px;
}
QLabel#reviewDialogTitle {
    color: #17201b;
    font-weight: 850;
    font-size: 22px;
}
QLabel#reviewCardGrade {
    background: #e1ebe3;
    color: #22543d;
    border-radius: 10px;
    padding: 4px 8px;
    font-weight: 800;
}
QLabel#reviewCardNature {
    color: #56625d;
    font-weight: 700;
}
QLabel#reviewCardCritique {
    color: #24312b;
    line-height: 1.4;
}
QLabel#reviewCardMetric {
    background: #f6f8f5;
    border: 1px solid #edf0ea;
    border-radius: 6px;
    padding: 7px;
    color: #34413a;
    font-weight: 650;
}
QTextEdit#reviewText {
    background: #ffffff;
    border: 1px solid #e1e4dd;
    border-radius: 8px;
    padding: 10px;
    color: #25322e;
    line-height: 1.45;
}
QHeaderView::section {
    background: #243447;
    color: #f8fafc;
    border: none;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    padding: 7px 8px;
    font-weight: 800;
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
