from __future__ import annotations

from pathlib import Path
import subprocess

try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
except ImportError as exc:  # pragma: no cover - only hit without desktop extras
    raise RuntimeError("生成 Qt 应用图标需要安装 PySide6：pip install PySide6") from exc


ICONSET_SPECS = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def build_macos_icon(icon_path: Path) -> Path:
    iconset_dir = icon_path.with_suffix(".iconset")
    iconset_dir.mkdir(parents=True, exist_ok=True)
    for filename, size in ICONSET_SPECS:
        _render_icon(iconset_dir / filename, size)
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["iconutil", "-c", "icns", "-o", str(icon_path), str(iconset_dir)], check=True)
    return icon_path


def create_app_icon(size: int = 256) -> QIcon:
    return QIcon(QPixmap.fromImage(_render_icon_image(size)))


def _render_icon(path: Path, size: int) -> None:
    _render_icon_image(size).save(str(path), "PNG")


def _render_icon_image(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = size * 0.08
    card = QRectF(margin, margin, size - margin * 2, size - margin * 2)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#18201d"))
    painter.drawRoundedRect(card, size * 0.18, size * 0.18)

    inner = card.adjusted(size * 0.18, size * 0.18, -size * 0.18, -size * 0.18)
    painter.setPen(QPen(QColor("#e8efe9"), max(2, int(size * 0.045)), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    points = (
        QPointF(inner.left(), inner.bottom() - inner.height() * 0.18),
        QPointF(inner.left() + inner.width() * 0.30, inner.bottom() - inner.height() * 0.46),
        QPointF(inner.left() + inner.width() * 0.55, inner.bottom() - inner.height() * 0.34),
        QPointF(inner.right(), inner.top() + inner.height() * 0.20),
    )
    for start, end in zip(points, points[1:]):
        painter.drawLine(start, end)

    painter.setBrush(QColor("#2d6a4f"))
    painter.setPen(Qt.PenStyle.NoPen)
    dot_radius = max(2, size * 0.045)
    for point in points:
        painter.drawEllipse(point, dot_radius, dot_radius)

    painter.end()
    return image
