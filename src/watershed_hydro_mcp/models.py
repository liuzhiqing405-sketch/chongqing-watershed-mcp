"""Request and record models for the hydro/stability MCP."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RunMode = Literal["preflight_only", "rainfall_only", "build_run_qa"]

#: The fixed stage whitelist. A client may name one of these to stop after;
#: it can never name a script, interpreter, executable or working directory.
WHITELIST_STAGES = (
    "sync", "validate", "delineate", "rainfall", "build",
    "rainfall_plot", "swmm", "safety", "analysis", "export", "qa",
)


class ParameterOverrides(BaseModel):
    """Stability parameters a run may override, bounded by the preset file."""

    deposit_thickness_m: float | None = None
    natural_density_g_cm3: float | None = None
    saturated_density_g_cm3: float | None = None
    saturated_cohesion_kpa: float | None = None
    saturated_friction_angle_deg: float | None = None
    initial_saturation: float | None = None
    friction_velocity_coefficient_a: float | None = None
    runoff_density_g_cm3: float | None = None
    calculation_resolution_m: float | None = None

    def present(self) -> dict[str, float]:
        return {key: value for key, value in self.model_dump().items() if value is not None}


class HydroRunRequest(BaseModel):
    input_id: str = Field(description="已导入或注册的输入编号")
    mode: RunMode = "build_run_qa"
    through: str | None = Field(
        default=None,
        description="在指定阶段后停止（含该阶段）；必须是白名单阶段名，空表示按模式跑完",
    )
    parameter_preset: str = "medium"
    parameter_overrides: ParameterOverrides = Field(default_factory=ParameterOverrides)
    notes: str = ""

    @field_validator("through")
    @classmethod
    def _check_stage(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in WHITELIST_STAGES:
            raise ValueError(f"未知阶段：{value}；允许值：{list(WHITELIST_STAGES)}")
        return value

    @field_validator("input_id")
    @classmethod
    def _check_input(cls, value: str) -> str:
        if not value.startswith("hydro-input-"):
            raise ValueError("输入编号必须以 hydro-input- 开头")
        return value

    @field_validator("notes")
    @classmethod
    def _check_notes(cls, value: str) -> str:
        return value.strip()[:2000]


class RegisterInputRequest(BaseModel):
    """Register data that came from somewhere other than the spatial MCP."""

    source_root: str = Field(
        description="包含待注册文件的目录；必须位于允许的收件目录内（exchange/imports 或 data）"
    )
    dem: str = Field(description="DEM 文件名，相对于 source_root")
    landuse: str = Field(description="土地利用 Shapefile 文件名（.shp），相对于 source_root")
    river: str | None = Field(default=None, description="水系 Shapefile 文件名，可选")
    boundary: str = Field(description="流域边界 GeoJSON 文件名，相对于 source_root")
    rainfall_zip: str = Field(description="站点降雨 ZIP 包文件名，相对于 source_root")
    labels: dict[str, str] = Field(default_factory=dict)


class VisualizationRequest(BaseModel):
    kind: str
    stage_id: str | None = None
    run_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
