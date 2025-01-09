# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ("C:\\Users\\bilel.guetarni\\Anaconda3\\envs\\work\\Lib\\site-packages\\dash", 'dash'),
        ("C:\\Users\\bilel.guetarni\\Anaconda3\\envs\\work\\Lib\\site-packages\\dash_bootstrap_components", 'dash_bootstrap_components'),
        ("C:\\Users\\bilel.guetarni\\Anaconda3\\envs\\work\\Lib\\site-packages\\dash_core_components", 'dash_core_components'),
        ("C:\\Users\\bilel.guetarni\\Anaconda3\\envs\\work\\Lib\\site-packages\\dash_daq", 'dash_daq'),
        ("C:\\Users\\bilel.guetarni\\Anaconda3\\envs\\work\\Lib\\site-packages\\dash_html_components", 'dash_html_components'),
        ("C:\\Users\\bilel.guetarni\\Anaconda3\\envs\\work\\Lib\\site-packages\\dash_table", 'dash_table'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
