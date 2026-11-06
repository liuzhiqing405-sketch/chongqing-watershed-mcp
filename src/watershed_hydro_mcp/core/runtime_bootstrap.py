from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLE = None
_GDAL_HANDLE = None


def configure_windows_geospatial_runtime() -> None:
    """Make Conda's GDAL runtime visible before rasterio is imported."""
    global _DLL_DIRECTORY_HANDLE, _GDAL_HANDLE

    if os.name != "nt":
        return

    prefix = Path(sys.prefix).resolve()
    local_prefix = (Path(__file__).resolve().parent / ".runtime").resolve()
    if prefix != local_prefix:
        return

    library_bin = prefix / "Library" / "bin"
    if not library_bin.is_dir():
        return

    gdal_data = prefix / "Library" / "share" / "gdal"
    proj_data = prefix / "Library" / "share" / "proj"
    if gdal_data.is_dir():
        os.environ.setdefault("GDAL_DATA", str(gdal_data))
    if proj_data.is_dir():
        os.environ.setdefault("PROJ_DATA", str(proj_data))
        os.environ.setdefault("PROJ_LIB", str(proj_data))

    os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLE = os.add_dll_directory(str(library_bin))

    gdal_dll = library_bin / "gdal.dll"
    if gdal_dll.is_file():
        _GDAL_HANDLE = ctypes.WinDLL(str(gdal_dll))


configure_windows_geospatial_runtime()
