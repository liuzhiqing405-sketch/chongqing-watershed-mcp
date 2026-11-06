from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


#: Project root: ``src/watershed_hydro_mcp/settings.py`` -> three levels up.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Filesystem layout for the hydro/stability MCP.

    Nothing here is shared with the spatial MCP: this service owns its own run
    root, input registry and job store, and reaches the other MCP only by
    reading a Model Input Package.
    """

    root: Path
    runs_root: Path
    hydro_root: Path
    native_root: Path
    config_root: Path

    @classmethod
    def from_env(cls) -> "Settings":
        root = PACKAGE_ROOT
        runs_root = Path(os.environ.get("WHS_RUNS_ROOT", root / "runs")).resolve()
        return cls(
            root=root,
            runs_root=runs_root,
            hydro_root=Path(os.environ.get("WHS_HYDRO_ROOT", runs_root / "hydro")).resolve(),
            native_root=Path(os.environ.get("WHS_NATIVE_ROOT", root / "native")).resolve(),
            config_root=Path(os.environ.get("WHS_CONFIG_ROOT", root / "config")).resolve(),
        )

    @property
    def inputs_root(self) -> Path:
        return self.hydro_root / "inputs"

    @property
    def jobs_root(self) -> Path:
        return self.hydro_root / "jobs"

    @property
    def visualizations_root(self) -> Path:
        return self.hydro_root / "visualizations"

    @property
    def comparisons_root(self) -> Path:
        return self.hydro_root / "comparisons"

    @property
    def engine(self) -> Path:
        return self.native_root / "swmm" / "runswmm.exe"

    @property
    def engine_present(self) -> bool:
        return self.engine.is_file() and (self.native_root / "swmm" / "swmm5.dll").is_file()

    def job_dir(self, run_id: str) -> Path:
        return self.jobs_root / run_id

    def input_dir(self, input_id: str) -> Path:
        return self.inputs_root / input_id

    def preset_file(self, name: str) -> Path:
        return self.config_root / name


SETTINGS = Settings.from_env()
