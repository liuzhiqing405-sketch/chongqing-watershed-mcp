"""Capability reporting: what this service can do right now, and what it cannot.

Reported before any run so a caller never has to discover a missing dependency
by failing a long computation.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import __version__
from .input_registry import inbox_roots, list_inputs
from .job_manager import MAXIMUM_CONCURRENT_HYDRO_RUNS
from .planner import MODE_STAGES, STAGES, check_engine, load_gates, load_presets
from .settings import SETTINGS, Settings


#: Modules every stage needs; reported so a broken environment is visible.
REQUIRED_MODULES = (
    "numpy", "scipy", "rasterio", "pyproj", "shapely", "shapefile", "yaml", "matplotlib",
)


def _module_status() -> dict[str, Any]:
    available: list[str] = []
    missing: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            available.append(f"{name} {getattr(module, '__version__', '')}".strip())
        except Exception as exc:  # noqa: BLE001 - report why it is unavailable
            missing[name] = str(exc)
    return {"available": available, "missing": missing,
            "ok": not missing}


def capabilities(settings: Settings = SETTINGS) -> dict[str, Any]:
    modules = _module_status()
    engine = check_engine(settings)
    presets = load_presets(settings)
    gates = load_gates(settings)
    inputs = list_inputs(settings)
    from .planner import _load_yaml
    from .grass_runtime import resolve_grass_cli
    defaults = _load_yaml(settings.preset_file('hydro.default.yaml'))
    method = (defaults.get('swmm') or {}).get('delineation_method')
    grass = resolve_grass_cli((defaults.get('executables') or {}).get('grass_cli'), settings.root)
    limitations = [
        "各子汇水区径流之和不是选定流域出口的路由流量过程：当前模型为独立子汇水区出水口配置。",
        "本服务不产生或修改流域边界、DEM、土地利用或水系；这些必须来自输入包或注册数据。",
        "SWMM 结果与安全系数结果是模型计算结果，需要通过质量门槛后才能作为结论使用。",
        "本服务与空间处理 MCP 相互独立，不会自动启动或调用对方。",
    ]
    if not engine["ok"]:
        limitations.append("SWMM 引擎文件不完整，建模与运行阶段不可用。")
    if modules["missing"]:
        limitations.append(f"缺少依赖：{sorted(modules['missing'])}")
    if method == 'grass_watershed' and grass is None:
        limitations.append('当前划分使用GRASS，但可执行程序缺失；任务将在规划阶段阻止启动。')
    return {
        "service": "watershed-hydro-stability-mcp",
        "version": __version__,
        "independent_of_spatial_mcp": True,
        "engine": engine,
        "delineation": {"method": method, "grass_available": grass is not None,
                        "grass_cli": str(grass) if grass else None,
                        "organization": "grass_basin_label" if method == 'grass_watershed' else 'native_d8',
                        "requires_single_exit": method == 'grass_watershed'},
        "modules": modules,
        "stages": [
            {"stage_id": stage.stage_id, "title": stage.title} for stage in STAGES
        ],
        "run_modes": {mode: list(stages) for mode, stages in MODE_STAGES.items()},
        "reportable_modes": ["preflight_only", "rainfall_only", "build_run_qa"],
        "parameter_presets": sorted((presets.get("presets") or {}).keys()),
        "parameter_bounds": presets.get("bounds") or {},
        "quality_gates": sorted((gates or {}).keys()),
        "concurrency": {"hydro_runs": MAXIMUM_CONCURRENT_HYDRO_RUNS, "visualization_jobs": 2},
        "registered_inputs": len(inputs),
        "inputs_root": str(settings.inputs_root),
        "inbox_roots": [str(path) for path in inbox_roots(settings)],
        "limitations": limitations,
        "next_actions": [
            "import_hydro_input_package 或 register_hydro_inputs 准备输入",
            "plan_hydro_run 查看将要执行的阶段与参数",
            "start_hydro_run 提交后台任务",
        ],
    }
