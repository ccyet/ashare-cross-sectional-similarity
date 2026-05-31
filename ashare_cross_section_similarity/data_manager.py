from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

KLINE_FILE_PATTERNS = ("*.parquet", "*.csv", "*.xlsx", "*.xls")
MIGRATION_COLUMNS = ["source", "destination", "status", "size_bytes", "message"]
MIGRATION_MODES = {"copy", "move"}


def plan_kline_migration(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    patterns: tuple[str, ...] = KLINE_FILE_PATTERNS,
) -> pd.DataFrame:
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    _validate_migration_paths(source_path, destination_path)

    files = _collect_kline_files(source_path, patterns)
    rows: list[dict[str, object]] = []
    for file_path in files:
        target_path = _target_path(source_path, destination_path, file_path)
        exists = target_path.exists()
        rows.append(
            {
                "source": str(file_path),
                "destination": str(target_path),
                "status": "ready" if overwrite or not exists else "exists",
                "size_bytes": int(file_path.stat().st_size),
                "message": "" if overwrite or not exists else "目标文件已存在",
            }
        )
    return pd.DataFrame(rows, columns=MIGRATION_COLUMNS)


def migrate_kline_data(
    source: str | Path,
    destination: str | Path,
    *,
    mode: str = "copy",
    overwrite: bool = False,
    patterns: tuple[str, ...] = KLINE_FILE_PATTERNS,
) -> pd.DataFrame:
    if mode not in MIGRATION_MODES:
        raise ValueError("迁移模式仅支持 copy 或 move。")

    plan = plan_kline_migration(source, destination, overwrite=overwrite, patterns=patterns)
    rows: list[dict[str, object]] = []
    for row in plan.to_dict("records"):
        source_path = Path(str(row["source"]))
        target_path = Path(str(row["destination"]))
        if row["status"] == "exists" and not overwrite:
            rows.append({**row, "status": "skipped"})
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if overwrite and target_path.exists():
                target_path.unlink()
            if mode == "copy":
                shutil.copy2(source_path, target_path)
                status = "copied"
            else:
                shutil.move(str(source_path), str(target_path))
                status = "moved"
        except OSError as exc:
            rows.append({**row, "status": "failed", "message": str(exc)})
            continue
        rows.append({**row, "status": status, "message": ""})

    if mode == "move":
        _remove_empty_source_dirs(Path(source).expanduser())
    return pd.DataFrame(rows, columns=MIGRATION_COLUMNS)


def _validate_migration_paths(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"来源路径不存在：{source}")
    if source.is_file() and not _matches_patterns(source, KLINE_FILE_PATTERNS):
        raise ValueError("来源文件不是支持的 K 线数据格式。")
    if source.is_file() and (destination / source.name).resolve(strict=False) == source.resolve(strict=True):
        raise ValueError("来源文件和目标文件不能相同。")
    if source.is_dir() and _is_relative_to(destination.resolve(strict=False), source.resolve(strict=True)):
        raise ValueError("目标目录不能位于来源目录内部。")
    if source.resolve(strict=True) == destination.resolve(strict=False):
        raise ValueError("来源路径和目标目录不能相同。")


def _collect_kline_files(source: Path, patterns: tuple[str, ...]) -> list[Path]:
    if source.is_file():
        return [source]
    files = [path for path in source.rglob("*") if path.is_file() and _matches_patterns(path, patterns)]
    return sorted(files, key=lambda path: str(path).lower())


def _target_path(source: Path, destination: Path, file_path: Path) -> Path:
    if source.is_file():
        return destination / source.name
    return destination / file_path.relative_to(source)


def _matches_patterns(path: Path, patterns: tuple[str, ...]) -> bool:
    lower_name = path.name.lower()
    return any(path.match(pattern) or lower_name.endswith(pattern.removeprefix("*").lower()) for pattern in patterns)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([path, parent]) == str(parent)
    except ValueError:
        return False


def _remove_empty_source_dirs(source: Path) -> None:
    if source.is_file() or not source.exists():
        return
    for directory in sorted((path for path in source.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
