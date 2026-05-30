from __future__ import annotations

from pathlib import Path
import subprocess
import sys

MACOS_APP_NAME = "A股相似阶段"
MACOS_BUNDLE_ID = "com.fincept.ashare-similarity"
WINDOWS_APP_NAME = "A股相似阶段"
WINDOWS_EXE_NAME = f"{WINDOWS_APP_NAME}.exe"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def macos_spec_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / "packaging" / "macos" / "ashare_xsec_sim_qt.spec"


def macos_icon_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / "build" / "macos" / f"{MACOS_APP_NAME}.icns"


def windows_spec_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / "packaging" / "windows" / "ashare_xsec_sim_qt.spec"


def windows_dist_path(root: Path | None = None) -> Path:
    base = project_root() if root is None else Path(root)
    return base / "dist" / "windows" / WINDOWS_APP_NAME / WINDOWS_EXE_NAME


def build_macos_command(root: Path | None = None) -> list[str]:
    return [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(macos_spec_path(root))]


def build_windows_command(root: Path | None = None) -> list[str]:
    base = project_root() if root is None else Path(root)
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(base / "dist" / "windows"),
        "--workpath",
        str(base / "build" / "windows"),
        str(windows_spec_path(base)),
    ]


def ensure_macos_platform(platform: str = sys.platform) -> None:
    if platform != "darwin":
        raise RuntimeError("macOS .app 打包只能在 macOS 上执行。")


def ensure_windows_platform(platform: str = sys.platform) -> None:
    if platform != "win32":
        raise RuntimeError("Windows .exe 打包只能在 Windows 上执行。")


def build_macos_app(*, root: Path | None = None, skip_icon: bool = False) -> Path:
    ensure_macos_platform()
    base = project_root() if root is None else Path(root)
    if not skip_icon:
        from ashare_cross_section_similarity.desktop.app_icon import build_macos_icon

        build_macos_icon(macos_icon_path(base))
    subprocess.run(build_macos_command(base), cwd=base, check=True)
    return base / "dist" / f"{MACOS_APP_NAME}.app"


def build_windows_app(*, root: Path | None = None) -> Path:
    ensure_windows_platform()
    base = project_root() if root is None else Path(root)
    subprocess.run(build_windows_command(base), cwd=base, check=True)
    return windows_dist_path(base)
