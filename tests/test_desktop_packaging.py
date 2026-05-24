from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from ashare_cross_section_similarity.desktop.packaging import (
    MACOS_APP_NAME,
    WINDOWS_EXE_NAME,
    build_macos_command,
    build_windows_command,
    ensure_macos_platform,
    ensure_windows_platform,
    macos_icon_path,
    macos_spec_path,
    windows_dist_path,
    windows_spec_path,
)


def test_macos_build_command_uses_repository_spec(tmp_path: Path) -> None:
    command = build_macos_command(tmp_path)

    assert command[:3] == [sys.executable, "-m", "PyInstaller"]
    assert "--clean" in command
    assert "--noconfirm" in command
    assert command[-1] == str(tmp_path / "packaging" / "macos" / "ashare_xsec_sim_qt.spec")


def test_macos_paths_are_stable_for_scripts_and_docs(tmp_path: Path) -> None:
    assert MACOS_APP_NAME == "A股相似阶段"
    assert macos_spec_path(tmp_path) == tmp_path / "packaging" / "macos" / "ashare_xsec_sim_qt.spec"
    assert macos_icon_path(tmp_path) == tmp_path / "build" / "macos" / "A股相似阶段.icns"


def test_windows_build_command_uses_repository_spec(tmp_path: Path) -> None:
    command = build_windows_command(tmp_path)

    assert command[:3] == [sys.executable, "-m", "PyInstaller"]
    assert "--clean" in command
    assert "--noconfirm" in command
    assert "--distpath" in command
    assert str(tmp_path / "dist" / "windows") in command
    assert "--workpath" in command
    assert str(tmp_path / "build" / "windows") in command
    assert command[-1] == str(tmp_path / "packaging" / "windows" / "ashare_xsec_sim_qt.spec")


def test_windows_paths_are_stable_for_scripts_and_docs(tmp_path: Path) -> None:
    assert WINDOWS_EXE_NAME == "A股相似阶段.exe"
    assert windows_spec_path(tmp_path) == tmp_path / "packaging" / "windows" / "ashare_xsec_sim_qt.spec"
    assert windows_dist_path(tmp_path) == tmp_path / "dist" / "windows" / "A股相似阶段" / "A股相似阶段.exe"


def test_macos_platform_guard_fails_explicitly_off_macos() -> None:
    with pytest.raises(RuntimeError, match="macOS"):
        ensure_macos_platform("linux")


def test_windows_platform_guard_fails_explicitly_off_windows() -> None:
    with pytest.raises(RuntimeError, match="Windows"):
        ensure_windows_platform("darwin")


def test_pyinstaller_spec_wraps_qt_entrypoint_and_bundle_name() -> None:
    spec = Path("packaging/macos/ashare_xsec_sim_qt.spec").read_text(encoding="utf-8")

    assert "ashare_cross_section_similarity/desktop/app.py" in spec
    assert "A股相似阶段.app" in spec
    assert "com.fincept.ashare-similarity" in spec
    assert "console=False" in spec


def test_pyinstaller_spec_excludes_non_desktop_research_stacks() -> None:
    spec = Path("packaging/macos/ashare_xsec_sim_qt.spec").read_text(encoding="utf-8")

    for module_name in ["PyQt5", "streamlit", "notebook", "sklearn", "stumpy", "aeon", "tslearn"]:
        assert f'"{module_name}"' in spec


def test_windows_pyinstaller_spec_wraps_qt_entrypoint_and_exe_name() -> None:
    spec = Path("packaging/windows/ashare_xsec_sim_qt.spec").read_text(encoding="utf-8")

    assert "ashare_cross_section_similarity/desktop/app.py" in spec
    assert "A股相似阶段.exe" in spec
    assert "console=False" in spec


def test_packaging_module_import_does_not_require_macos_icon_dependencies() -> None:
    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "ashare_cross_section_similarity.desktop.app_icon":
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from ashare_cross_section_similarity.desktop import packaging
print(packaging.WINDOWS_EXE_NAME)
"""
    completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert completed.stdout.strip() == "A股相似阶段.exe"


def test_build_scripts_prefer_repository_source_over_installed_wheel() -> None:
    for script_path in [Path("scripts/build_macos_qt_app.py"), Path("scripts/build_windows_qt_app.py")]:
        script = script_path.read_text(encoding="utf-8")

        assert "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))" in script
