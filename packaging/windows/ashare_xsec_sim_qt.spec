# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


ROOT = Path(SPECPATH).resolve().parents[1]
APP_NAME = "A股相似阶段"
EXE_NAME = "A股相似阶段.exe"
AKSHARE_DATAS = collect_data_files("akshare")


a = Analysis(
    [str(ROOT / "ashare_cross_section_similarity/desktop/app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=AKSHARE_DATAS,
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "pyarrow",
        "pyarrow.parquet",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "aeon",
        "altair",
        "astropy",
        "black",
        "bokeh",
        "botocore",
        "dask",
        "distributed",
        "duckdb",
        "h5py",
        "IPython",
        "jupyterlab",
        "matplotlib",
        "nbconvert",
        "nbformat",
        "notebook",
        "openbb",
        "panel",
        "plotly",
        "pytest",
        "skimage",
        "sklearn",
        "sphinx",
        "statsmodels",
        "streamlit",
        "stumpy",
        "tables",
        "torch",
        "tslearn",
        "xarray",
        "yapf",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
