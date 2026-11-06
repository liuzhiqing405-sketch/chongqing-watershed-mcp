from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _load_defaults(root: Path) -> dict[str, Any]:
    path = Path(os.environ.get("CQ_CONFIG", root / "config" / "spatial.default.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"缺少空间处理默认配置：{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@dataclass(frozen=True)
class Settings:
    """Filesystem layout and thresholds for the spatial MCP.

    Every path is derived from the project root or an environment override, so
    the project runs from any directory without a drive letter being assumed.
    Filenames and thresholds come from ``config/spatial.default.yaml``.
    """

    root: Path
    data_root: Path
    runs_root: Path
    exchange_root: Path
    native_root: Path
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        root = PACKAGE_ROOT
        return cls(
            root=root,
            data_root=Path(os.environ.get("CQ_DATA_ROOT", root / "data")).resolve(),
            runs_root=Path(os.environ.get("CQ_RUNS_ROOT", root / "runs")).resolve(),
            exchange_root=Path(os.environ.get("CQ_EXCHANGE_ROOT", root / "exchange")).resolve(),
            native_root=Path(os.environ.get("CQ_NATIVE_ROOT", root / "native")).resolve(),
            defaults=_load_defaults(root),
        )

    # ------------------------------------------------------------ configuration
    @property
    def datasets(self) -> dict[str, str]:
        return self.defaults.get("datasets") or {}

    @property
    def delineation(self) -> dict[str, Any]:
        return self.defaults.get("delineation") or {}

    @property
    def rainfall_defaults(self) -> dict[str, Any]:
        return self.defaults.get("rainfall") or {}

    @property
    def vector_fallback_encoding(self) -> str:
        return str(self.defaults.get("vector_fallback_encoding") or "gbk")

    def dataset(self, key: str) -> Path:
        name = self.datasets.get(key)
        if not name:
            raise KeyError(f"config/spatial.default.yaml 未定义数据集：{key}")
        return self.data_root / name

    # ----------------------------------------------------------- managed data
    @property
    def dem(self) -> Path:
        return self.dataset("dem")

    @property
    def rivers(self) -> Path:
        return self.dataset("rivers")

    @property
    def display_rivers(self) -> Path | None:
        """Optional reference network, separate from the named input layer."""
        return self.dataset("display_rivers") if self.datasets.get("display_rivers") else None

    @property
    def display_rivers_label(self) -> str:
        return str((self.defaults.get("map_display") or {}).get("river_label") or "参考河网")

    @property
    def landuse(self) -> Path:
        return self.dataset("landuse")

    @property
    def boundary(self) -> Path:
        return self.dataset("boundary")

    @property
    def rainfall_root(self) -> Path:
        return self.data_root / (self.datasets.get("rainfall_directory") or "water/逐时降水_宽表")

    @property
    def rainfall_pattern(self) -> str:
        return self.datasets.get("rainfall_pattern") or "ERA5_重庆逐时降水_*.csv"

    # ----------------------------------------------------------------- outputs
    @property
    def watersheds_root(self) -> Path:
        return self.runs_root / "watersheds"

    @property
    def rainfall_runs_root(self) -> Path:
        return self.runs_root / "rainfall"

    @property
    def model_inputs_root(self) -> Path:
        return self.runs_root / "model_inputs"

    @property
    def spatial_visualizations_root(self) -> Path:
        """Where rendered process figures are collected, one folder per review."""
        return self.runs_root / "spatial_visualizations"

    @property
    def exports_root(self) -> Path:
        return self.exchange_root / "exports"

    @property
    def config_path(self) -> Path:
        return Path(os.environ.get("CQ_CONFIG", self.root / "config" / "spatial.default.yaml"))


SETTINGS = Settings.from_env()
