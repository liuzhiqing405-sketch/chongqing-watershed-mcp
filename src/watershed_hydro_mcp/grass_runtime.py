"""Resolve a trusted, locally configured GRASS installation."""
from __future__ import annotations

import os
import locale
from pathlib import Path
import shutil
import subprocess
import tempfile


def resolve_grass_cli(configured=None, root: Path | None = None) -> Path | None:
    candidates = [str(configured)] if configured else ["grass", "grass85", "grass84", "grass83"]
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return Path(found).resolve()
        path = Path(candidate).expanduser()
        if not path.is_absolute() and root:
            path = root / path
        if path.is_file():
            return path.resolve()
    return None


GRASS_OUTPUTS = ("basin.tif", "drainage.tif", "accumulation.tif", "stream.tif", "version.txt")


def _decode_output(value: bytes | str | None) -> str:
    if isinstance(value, str):
        return value
    for encoding in ("utf-8", locale.getencoding()):
        try:
            return (value or b"").decode(encoding)
        except UnicodeDecodeError:
            continue
    return (value or b"").decode("utf-8", errors="replace")


def _grass_path(path: Path) -> Path:
    """Windows GRASS/GDAL native tools require an ASCII workspace path."""
    if os.name != "nt" or str(path).isascii():
        return path
    import ctypes
    from ctypes import wintypes

    get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short_path.restype = wintypes.DWORD
    size = get_short_path(str(path), None, 0)
    if size:
        buffer = ctypes.create_unicode_buffer(size)
        length = get_short_path(str(path), buffer, size)
        if 0 < length < size and buffer.value.isascii():
            return Path(buffer.value)
    raise RuntimeError("GRASS requires an ASCII temporary directory on Windows; configure TEMP/TMP to an ASCII path")


def run_grass(cli: Path, dem: Path, worker: Path, output: Path, threshold: int):
    env = os.environ.copy()
    # QGIS supplies its own Python runtime. A virtualenv's Python settings
    # otherwise prevent grass.script and the GDAL modules from loading.
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    output.mkdir(parents=True, exist_ok=True)
    # g.proj/r.in.gdal cannot open Unicode filenames in some Windows builds.
    # Stage the unchanged DEM and worker, and export to the same ASCII workspace.
    # This also keeps concurrently running jobs in separate GRASS databases.
    with tempfile.TemporaryDirectory(prefix="watershed-grass-") as temporary:
        workspace = _grass_path(Path(temporary))
        local_dem, local_worker = workspace / "dem.tif", workspace / "worker.py"
        local_output = workspace / "output"
        local_output.mkdir()
        shutil.copy2(dem, local_dem)
        shutil.copy2(worker, local_worker)
        # GRASS creates its own temporary project; keep that path ASCII as well.
        env["TMPDIR"] = env["TEMP"] = env["TMP"] = str(workspace)
        command = [str(cli), "--tmp-project", str(local_dem), "--exec", "python",
                   str(local_worker), str(local_dem), str(local_output), str(threshold)]
        try:
            result = subprocess.run(command, env=env, capture_output=True, timeout=3600,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired as exc:
            (output / "execution.log").write_text(
                _decode_output(exc.stdout) + "\n" + _decode_output(exc.stderr) + "\nGRASS timed out after 3600 seconds\n",
                encoding="utf-8")
            raise
        (output / "execution.log").write_text(
            _decode_output(result.stdout) + "\n" + _decode_output(result.stderr), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"GRASS r.watershed failed ({result.returncode}); see {output / 'execution.log'}")
        missing = [name for name in GRASS_OUTPUTS if not (local_output / name).is_file()]
        if missing:
            raise RuntimeError(f"GRASS did not produce required outputs: {', '.join(missing)}; see {output / 'execution.log'}")
        for name in GRASS_OUTPUTS:
            shutil.copy2(local_output / name, output / name)

