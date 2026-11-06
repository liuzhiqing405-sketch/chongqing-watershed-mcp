"""Run a generated small basin through the hydrology job manager."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import build_package
from integration.test_integration import wait_for
from watershed_hydro_mcp.input_registry import import_package
from watershed_hydro_mcp.job_manager import JobStore
from watershed_hydro_mcp.models import HydroRunRequest
from watershed_hydro_mcp.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    work = args.output.resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        parser.error("--output must be a new or empty directory")

    installed = Settings.from_env()
    if not installed.engine_present:
        parser.error(f"install EPA SWMM 5.2.4 under {installed.native_root / 'swmm'}")

    runs = work / "runs"
    settings = Settings(
        root=work,
        runs_root=runs,
        hydro_root=runs / "hydro",
        native_root=installed.native_root,
        config_root=ROOT / "config",
    )
    package = build_package(work / "exchange" / "exports" / "package-abcdef000001")
    record = import_package(package.name, settings)
    store = JobStore(settings)
    started = store.start(HydroRunRequest(input_id=record["input_id"], mode="build_run_qa"))
    run_id = started["run_id"]
    job = settings.job_dir(run_id)
    final = wait_for(store, run_id, timeout=900)
    print(f"Status: {final['status']}")
    print(f"Job: {job}")
    if final["status"] not in {"succeeded", "qa_failed"}:
        raise SystemExit(final.get("summary") or "Synthetic case failed")

    folder = job / "project" / "rapid_safety_factor_results"
    raster, = folder.glob("*_enhanced_minimum_safety_factor.tif")
    with rasterio.open(raster) as dataset:
        values = dataset.read(1, masked=True)
        cell_area = abs(dataset.transform.determinant)
        unstable_area = np.count_nonzero(values.compressed() < 1) * cell_area
    print(f"Minimum factor of safety: {values.min():.6f}")
    print(f"Modeled area with minimum factor of safety < 1: {unstable_area:.2f} m²")
    print(f"Raster: {raster}")
    if final["status"] == "qa_failed":
        raise SystemExit("See the job directory for the quality-check results.")


if __name__ == "__main__":
    main()
