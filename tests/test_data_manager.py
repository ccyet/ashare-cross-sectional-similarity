from __future__ import annotations

from pathlib import Path

import pytest

from ashare_cross_section_similarity.data_manager import migrate_kline_data, plan_kline_migration


def test_plan_kline_migration_preserves_directory_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    kline_file = source / "daily" / "qfq" / "000001.SZ.parquet"
    ignored_file = source / "daily" / "qfq" / "note.txt"
    kline_file.parent.mkdir(parents=True)
    kline_file.write_text("bars")
    ignored_file.write_text("ignore")

    plan = plan_kline_migration(source, destination)

    assert plan["source"].tolist() == [str(kline_file)]
    assert plan["destination"].tolist() == [str(destination / "daily" / "qfq" / "000001.SZ.parquet")]
    assert plan["status"].tolist() == ["ready"]


def test_migrate_kline_data_copies_without_overwriting(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    kline_file = source / "000001.SZ.parquet"
    target_file = destination / "000001.SZ.parquet"
    source.mkdir()
    destination.mkdir()
    kline_file.write_text("new")
    target_file.write_text("old")

    result = migrate_kline_data(source, destination, mode="copy", overwrite=False)

    assert result.loc[0, "status"] == "skipped"
    assert target_file.read_text() == "old"
    assert kline_file.exists()


def test_migrate_kline_data_moves_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    kline_file = source / "000001.SZ.parquet"
    source.mkdir()
    kline_file.write_text("bars")

    result = migrate_kline_data(kline_file, destination, mode="move")

    assert result.loc[0, "status"] == "moved"
    assert not kline_file.exists()
    assert (destination / "000001.SZ.parquet").read_text() == "bars"


def test_plan_kline_migration_rejects_destination_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="目标目录不能位于来源目录内部"):
        plan_kline_migration(source, source / "nested")


def test_plan_kline_migration_rejects_file_target_same_as_source(tmp_path: Path) -> None:
    source = tmp_path / "000001.SZ.parquet"
    source.write_text("bars")

    with pytest.raises(ValueError, match="来源文件和目标文件不能相同"):
        plan_kline_migration(source, tmp_path)
