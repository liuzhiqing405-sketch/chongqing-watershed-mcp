"""Plan a hydro run: lay out its own project directory and write its config.

Every run gets a private project directory seeded from the managed input. The
configuration written here is the only place paths are decided, and the client
never supplies a script name, interpreter, working directory or executable: the
stage whitelist below is the whole vocabulary.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
import rasterio
from watershed_exchange.dem_budget import DEFAULT_MAXIMUM_PROCESSING_CELLS, cell_budget, require_cells

from .settings import SETTINGS, Settings


@dataclass(frozen=True)
class Stage:
    stage_id: str
    title: str
    module: str


#: The fixed pipeline. A client may name a stage to stop after; it may never
#: name a file, an interpreter or a command.
STAGES: tuple[Stage, ...] = (
    Stage("sync", "根据事件同步配置", "watershed_hydro_mcp.core.sync"),
    Stage("validate", "输入检查", "watershed_hydro_mcp.core.validate"),
    Stage("delineate", "由DEM自动提取SWMM分区", "watershed_hydro_mcp.core.delineation.delineate"),
    Stage("rainfall", "准备站点空间降雨", "watershed_hydro_mcp.core.rainfall.prepare_rainfall"),
    Stage("build", "生成SWMM输入与子汇水区降雨", "watershed_hydro_mcp.core.swmm.build_inp"),
    Stage("rainfall_plot", "绘制多站降雨过程图", "watershed_hydro_mcp.core.rainfall.plot_series"),
    Stage("swmm", "运行SWMM", "watershed_hydro_mcp.core.swmm.run_swmm"),
    Stage("safety", "生成安全系数与水力要素结果", "watershed_hydro_mcp.core.stability.safety_factor"),
    Stage("analysis", "分析参数敏感性、贡献与安全系数时序", "watershed_hydro_mcp.core.sensitivity.analyze"),
    Stage("export", "输出最终结果文件夹", "watershed_hydro_mcp.core.export.final_results"),
    Stage("qa", "质量检查", "watershed_hydro_mcp.qa"),
)

STAGE_IDS: tuple[str, ...] = tuple(stage.stage_id for stage in STAGES)

#: Stage subsets per run mode.
MODE_STAGES: dict[str, tuple[str, ...]] = {
    "preflight_only": ("sync", "validate"),
    "rainfall_only": ("sync", "validate", "delineate", "rainfall", "rainfall_plot"),
    "build_run_qa": STAGE_IDS,
}

#: Files the run project needs, as (source-relative-name, destination-relative).
PROJECT_INPUTS: tuple[tuple[str, str], ...] = (
    ("dem_context.tif", "inputs/user_data/dem_context.tif"),
    ("dem.tif", "inputs/user_data/dem.tif"),
    ("station_rainfall.zip", "inputs/user_data/station_rainfall.zip"),
    ("basin.geojson", "inputs/user_data/basin.geojson"),
    ("landuse.shp", "inputs/user_data/landuse.shp"),
    ("landuse.shx", "inputs/user_data/landuse.shx"),
    ("landuse.dbf", "inputs/user_data/landuse.dbf"),
    ("landuse.prj", "inputs/user_data/landuse.prj"),
    ("landuse.cpg", "inputs/user_data/landuse.cpg"),
    ("river.shp", "inputs/user_data/river.shp"),
    ("river.shx", "inputs/user_data/river.shx"),
    ("river.dbf", "inputs/user_data/river.dbf"),
    ("river.prj", "inputs/user_data/river.prj"),
    ("river.cpg", "inputs/user_data/river.cpg"),
)

#: Directories a run project owns, so every stage has somewhere to write.
PROJECT_DIRECTORIES: tuple[str, ...] = (
    "inputs/user_data",
    "inputs/rainfall",
    "inputs/dem_derived",
    "inputs/rainfall/figures",
    "swmm/model",
    "swmm/results",
    "rapid_safety_factor_results",
    "analysis_results",
    "final_results",
)


class PlanningError(ValueError):
    """Raised when a run cannot be planned from the given input and settings."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlanningError(f"缺少配置文件：{path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_presets(settings: Settings = SETTINGS) -> dict[str, Any]:
    return _load_yaml(settings.preset_file("stability_presets.yaml"))


def load_gates(settings: Settings = SETTINGS) -> dict[str, Any]:
    return _load_yaml(settings.preset_file("quality_gates.yaml"))


def _slug(value: str) -> str:
    from .core.common import slugify

    return slugify(value)


def _parse(value: str, label: str) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise PlanningError(f"{label} 时间格式无法解析：{value}")


def _event_times(record: dict[str, Any]) -> tuple[datetime, datetime, datetime]:
    event = record.get("event") or {}
    start = event.get("start_datetime")
    end = event.get("end_datetime")
    if start and end:
        event_start = _parse(start, "事件开始")
        event_end = _parse(end, "事件结束")
    else:
        # A registered input may carry no event metadata; fall back to one full
        # day so the pipeline still has a well-formed time window.
        event_start = datetime.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        event_end = event_start.replace(hour=23)
    if event_end < event_start:
        raise PlanningError("事件结束时间早于开始时间")
    hours = (event_end - event_start).total_seconds() / 3600.0
    if hours > 246:
        raise PlanningError(f"事件长达 {hours:.0f} 小时，超过雨站包支持的 247 个时次")
    simulation_end = _parse(str(event["simulation_end_datetime"]), "模拟结束") if event.get(
        "simulation_end_datetime"
    ) else event_end + timedelta(hours=2)
    return event_start, event_end, simulation_end


def _override_stability(
    defaults: dict[str, Any], preset_name: str, overrides: dict[str, float], settings: Settings
) -> dict[str, Any]:
    presets = load_presets(settings)
    available = presets.get("presets") or {}
    if preset_name not in available:
        raise PlanningError(f"未知参数预设：{preset_name}；可用：{sorted(available)}")
    stability = dict(defaults.get("stability") or {})
    stability.update(available[preset_name])
    stability["parameter_preset"] = preset_name
    bounds = presets.get("bounds") or {}
    for key, value in overrides.items():
        if key in bounds:
            low, high = bounds[key]
            if not (low <= float(value) <= high):
                raise PlanningError(f"参数 {key}={value} 超出允许范围 [{low}, {high}]")
        stability[key] = value
    return stability


def terrain_preflight(record: dict[str, Any], settings: Settings = SETTINGS) -> dict[str, Any]:
    """Check headers before copying inputs or approving a runnable plan."""
    defaults = _load_yaml(settings.preset_file("hydro.default.yaml"))
    declared = ((record.get("provenance") or {}).get("dem_processing") or {}).get("maximum_cells")
    configured = (defaults.get("terrain") or {}).get("maximum_processing_cells")
    limits = [cell_budget(value) for value in (declared, configured) if value is not None]
    maximum = min(limits) if limits else DEFAULT_MAXIMUM_PROCESSING_CELLS
    facts = {"maximum_processing_cells": maximum, "grids": {}}
    facts['delineation'] = {
        'method': (defaults.get('swmm') or {}).get('delineation_method'),
        'channel_threshold_ha': (defaults.get('swmm') or {}).get('channel_threshold_ha', 20),
        'outlet_selection': (defaults.get('swmm') or {}).get('outlet_selection', 'drainage_exit'),
    }
    if (defaults.get("swmm") or {}).get("delineation_method") == "grass_watershed":
        facts['delineation'].update(organization='grass_basin_label', requires_single_exit=True)
        from .grass_runtime import resolve_grass_cli
        grass = resolve_grass_cli((defaults.get("executables") or {}).get("grass_cli"), settings.root)
        if grass is None:
            raise PlanningError("GRASS划分需要 GRASS GIS；未找到可执行程序，请配置 executables.grass_cli")
        facts["grass_cli"] = str(grass)

    for name in ("dem.tif", "dem_context.tif"):
        path = settings.input_dir(record["input_id"]) / name
        if name == "dem_context.tif" and not path.is_file():
            continue
        with rasterio.open(path) as source:
            cells = source.width*source.height
            require_cells(cells, maximum, name)
            facts["grids"][name] = {"width": source.width, "height": source.height, "cells": cells}
    return facts


def build_project(
    run_id: str,
    record: dict[str, Any],
    request,
    settings: Settings = SETTINGS,
) -> dict[str, Any]:
    """Create the run's private project directory and its ``config.yaml``."""
    job_dir = settings.job_dir(run_id)
    project = job_dir / "project"
    if project.exists():
        raise PlanningError(f"运行目录已存在：{run_id}")
    source = settings.input_dir(record["input_id"])
    terrain = terrain_preflight(record, settings)
    project.mkdir(parents=True)
    for relative in PROJECT_DIRECTORIES:
        (project / relative).mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name, destination in PROJECT_INPUTS:
        origin = source / name
        if not origin.is_file():
            if name.startswith("river.") or name == "dem_context.tif":
                continue
            shutil.rmtree(project, ignore_errors=True)
            raise PlanningError(f"输入包缺少文件：{name}")
        target = project / destination
        shutil.copy2(origin, target)
        copied.append(destination)

    defaults = _load_yaml(settings.preset_file("hydro.default.yaml"))
    event_start, event_end, simulation_end = _event_times(record)
    slug = _slug(str((record.get("source_package_id") or record["input_id"])))
    stability = _override_stability(
        defaults, request.parameter_preset, request.parameter_overrides.present(), settings
    )

    def absolute(relative: str) -> str:
        return str((project / relative).resolve()).replace("\\", "/")

    config: dict[str, Any] = {
        "project": {"name": slug, "root": str(project.resolve()).replace("\\", "/")},
        "event": {
            "name": slug,
            "date": event_start.strftime("%Y-%m-%d"),
            "start_datetime": event_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_datetime": event_end.strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_end_datetime": simulation_end.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "executables": {
            "grass_cli": terrain.get("grass_cli"),
            "swmm_cli": str(settings.engine.resolve()).replace("\\", "/") if settings.engine_present else None,
        },
        "study_area": {
            **(defaults.get("study_area") or {}),
            "name": str((record.get("study_area") or {}).get("crs") or slug),
            "boundary_shp": None,
            "landuse_shp": absolute("inputs/user_data/landuse.shp"),
            "river_shp": absolute("inputs/user_data/river.shp"),
        },
        "terrain": {"maximum_processing_cells": terrain["maximum_processing_cells"],
                    "dem": absolute("inputs/user_data/dem.tif"),
                    "drainage_context_dem": absolute("inputs/user_data/dem_context.tif")
                    if (project / "inputs/user_data/dem_context.tif").is_file() else None},
        "rainfall": {
            **(defaults.get("rainfall") or {}),
            "station_zip": absolute("inputs/user_data/station_rainfall.zip"),
        },
        "swmm": dict(defaults.get("swmm") or {}),
        "stability": stability,
        "analysis": dict(defaults.get("analysis") or {}),
    }
    (project / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    plan = {
        "run_id": run_id,
        "project": str(project),
        "config": str(project / "config.yaml"),
        "slug": slug,
        "event": config["event"],
        "copied_inputs": copied,
        "engine": config["executables"]["swmm_cli"],
        "engine_present": settings.engine_present,
        "parameter_preset": request.parameter_preset,
        "parameter_overrides": request.parameter_overrides.present(),
        "dem_preflight": terrain,
    }
    return plan


def stages_for(mode: str, through: str | None) -> list[Stage]:
    selected = MODE_STAGES.get(mode)
    if selected is None:
        raise PlanningError(f"未知运行模式：{mode}")
    stages = [stage for stage in STAGES if stage.stage_id in selected]
    if through is None:
        return stages
    if through not in selected:
        raise PlanningError(f"阶段 {through} 不适用于模式 {mode}")
    index = next(i for i, stage in enumerate(stages) if stage.stage_id == through)
    return stages[: index + 1]


def check_engine(settings: Settings = SETTINGS) -> dict[str, Any]:
    """Report whether the bundled SWMM engine is usable without starting it."""
    files = {
        name: (settings.native_root / "swmm" / name).is_file()
        for name in ("runswmm.exe", "swmm5.dll", "vcomp140.dll")
    }
    return {"ok": all(files.values()), "files": files,
            "directory": str(settings.native_root / "swmm")}
