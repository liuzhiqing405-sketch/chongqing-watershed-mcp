from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml


from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT, engine_executable

CFG = LazyConfig()


def find_engine() -> Path:
    candidates = [
        os.environ.get("PIPELINE_SWMM_CLI"),
        CFG.get("executables", {}).get("swmm_cli"),
        engine_executable(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("The runoff engine was not found. Keep the bundled native/swmm engine with the program.")


def main() -> None:
    exe = find_engine()
    inp = Path(os.environ.get("PIPELINE_SWMM_INP") or CFG["swmm"]["working_inp"])
    rpt = Path(os.environ.get("PIPELINE_SWMM_RPT") or CFG["swmm"]["report"])
    out = Path(os.environ.get("PIPELINE_SWMM_OUT") or CFG["swmm"]["output"])
    rpt.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    if rpt.exists():
        rpt.unlink()

    cmd = [str(exe), str(inp), str(rpt), str(out)]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print("Command:", " ".join(cmd))
    print("Return code:", result.returncode)
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)

    if not rpt.exists():
        raise RuntimeError(f"SWMM report was not written: {rpt}")
    report_text = rpt.read_text(encoding="utf-8", errors="replace")
    error_lines = []
    for line in report_text.splitlines():
        upper = line.upper().lstrip()
        if upper.startswith("ERROR ") or upper.startswith("ERROR:") or upper.startswith("*** ERROR"):
            error_lines.append(line)
    warning_lines = [line for line in report_text.splitlines() if "WARNING" in line.upper()]
    print(f"Report: {rpt}")
    print(f"Output: {out}")
    print(f"Errors: {len(error_lines)}")
    print(f"Warnings: {len(warning_lines)}")
    for line in error_lines[:20]:
        print(line)

    if result.returncode != 0 or error_lines:
        raise SystemExit(1)
    from .landuse_response import write_parent_runoff
    write_parent_runoff(out, inp.parent)


if __name__ == "__main__":
    main()
