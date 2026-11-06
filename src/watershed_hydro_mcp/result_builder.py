"""Summaries of what a run produced, and inspection of its SWMM model.

Text is parsed rather than trusted: the model report is the source of truth for
continuity errors and warnings, and every summary carries the caveat that this
model uses independent subcatchment outfalls.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from watershed_exchange.validation import read_text_limited

from .artifact_registry import load_artifacts
from .qa import read_gates
from .state import read_json


#: Each drainage partition has an outfall shared by its land-use units.
#: No conduit network is built, so summed runoff is not an outlet hydrograph.
ROUTING_LIMITATION = "各子汇水区径流之和不是选定流域出口的路由流量过程。"

TEXT_SUFFIXES = {".txt", ".log", ".csv", ".md", ".json", ".inp", ".rpt"}


def _sections(inp_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in inp_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.strip("[]").upper()
            sections.setdefault(current, [])
            continue
        if current and stripped and not stripped.startswith(";;"):
            sections[current].append(stripped)
    return sections


def _rows(lines: list[str]) -> list[list[str]]:
    return [line.split() for line in lines if line.split()]


def inspect_swmm_model(project: Path) -> dict[str, Any]:
    inp_path = _first(project / "swmm/model", "*.inp")
    report_path = _first(project / "swmm/results", "*.rpt")
    if inp_path is None:
        return {"available": False, "reason": "尚未生成 SWMM 输入文件",
                "limitation": ROUTING_LIMITATION}
    text = inp_path.read_text(encoding="utf-8", errors="replace")
    sections = _sections(text)
    subcatchments = _rows(sections.get("SUBCATCHMENTS", []))
    outfalls = _rows(sections.get("OUTFALLS", []))
    gages = _rows(sections.get("RAINGAGES", []))
    timeseries = _rows(sections.get("TIMESERIES", []))
    junctions = _rows(sections.get("JUNCTIONS", []))
    conduits = _rows(sections.get("CONDUITS", []))

    areas = [_float(row[3]) for row in subcatchments if len(row) > 4]
    areas = [value for value in areas if value == value]
    imperviousness = [_float(row[4]) for row in subcatchments if len(row) > 4]
    imperviousness = [value for value in imperviousness if value == value]

    options = _options(sections.get("OPTIONS", []))
    model = {
        "available": True,
        "inp_file": inp_path.name,
        "subcatchment_count": len(subcatchments),
        "raingage_count": len(gages),
        "junction_count": len(junctions),
        "conduit_count": len(conduits),
        "outfall_count": len(outfalls),
        "timeseries_rows": len(timeseries),
        "area_ha": {
            "min": round(min(areas), 4) if areas else None,
            "max": round(max(areas), 4) if areas else None,
            "total": round(sum(areas), 4) if areas else None,
        },
        "impervious_percent": {
            "min": round(min(imperviousness), 3) if imperviousness else None,
            "max": round(max(imperviousness), 3) if imperviousness else None,
        },
        "simulation_window": {
            "start": f"{options.get('START_DATE', '')} {options.get('START_TIME', '')}".strip(),
            "end": f"{options.get('END_DATE', '')} {options.get('END_TIME', '')}".strip(),
            "report_step": options.get("REPORT_STEP"),
            "wet_step": options.get("WET_STEP"),
            "routing_step": options.get("ROUTING_STEP"),
        },
        "routing_network_present": bool(junctions or conduits),
        "limitation": ROUTING_LIMITATION,
    }
    if "; LANDUSE_REPRESENTATION response_units" in text:
        parent_path = inp_path.parent / "auto_swmm_parameter_review.csv"
        parents = _read_csv(parent_path) if parent_path.is_file() else []
        model["landuse_representation"] = "response_units"
        model["internal_response_unit_count"] = len(subcatchments)
        model["subcatchment_count"] = len(parents)
        parent_areas = [_float(row.get("area_ha")) for row in parents]
        if parent_areas:
            model["area_ha"].update(min=round(min(parent_areas),4), max=round(max(parent_areas),4))
            model["impervious_percent"].update(
                min=round(min(_float(row.get("imperv_pct")) for row in parents),3),
                max=round(max(_float(row.get("imperv_pct")) for row in parents),3))
        area_review = inp_path.parent / "auto_swmm_landuse_area_review.csv"
        if area_review.is_file():
            model["landuse_area_review"] = _read_csv(area_review)
    if report_path is not None:
        report_text = report_path.read_text(encoding="utf-8", errors="replace")
        model["report"] = {
            "file": report_path.name,
            "errors": [
                line.strip() for line in report_text.splitlines()
                if line.upper().lstrip().startswith(("ERROR ", "ERROR:", "*** ERROR"))
            ][:20],
            "warnings": [line.strip() for line in report_text.splitlines() if "WARNING" in line.upper()][:20],
            "continuity": _continuity(report_text),
            "figures_available": _figures(project),
        }
    else:
        model["report"] = {"file": None, "errors": [], "warnings": [],
                           "continuity": {}, "figures_available": _figures(project)}
    return model


def _options(lines: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for row in _rows(lines):
        if len(row) >= 2:
            options[row[0].upper()] = row[1]
    return options


def _continuity(report_text: str) -> dict[str, Any]:
    """Read the continuity error blocks SWMM writes after a run."""
    result: dict[str, Any] = {}
    blocks = re.finditer(
        r"(Runoff Quantity Continuity|Flow Routing Continuity)([\s\S]{0,1200}?)(?:\n\s*\n|\Z)",
        report_text,
    )
    for block in blocks:
        title = block.group(1)
        body = block.group(2)
        entry: dict[str, Any] = {}
        match = re.search(r"Continuity Error \(%\)\s*\.*\s*(-?\d+\.?\d*)", body)
        if match:
            entry["continuity_error_percent"] = float(match.group(1))
        for label, key in (
            ("Total Precipitation", "total_precipitation"),
            ("Total Runoff", "total_runoff"),
            ("Peak Runoff", "peak_runoff"),
            ("Total Inflow", "total_inflow"),
            ("Total Outflow", "total_outflow"),
        ):
            found = re.search(rf"{label}[^\d\-]*(-?\d[\d.,eE+\-]*)", body)
            if found:
                entry[key] = found.group(1)
        result["runoff" if "Runoff" in title else "routing"] = entry
    return result


def _figures(project: Path) -> list[str]:
    directory = project / "swmm/visual"
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob("*.png"))


def _first(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def safety_summary(project: Path) -> dict[str, Any]:
    summary = _first(project / "rapid_safety_factor_results", "*_safety_factor_summary.csv")
    if summary is None:
        return {"available": False, "reason": "尚未生成安全系数结果"}
    rows = {row.get("metric"): row.get("value") for row in _read_csv(summary)}
    result = {
        "available": True,
        "file": summary.name,
        "minimum_safety_factor": _number(rows.get("minimum_safety_factor_min")),
        "mean_safety_factor": _number(rows.get("minimum_safety_factor_mean")),
        "p95_safety_factor": _number(rows.get("minimum_safety_factor_p95")),
        "maximum_safety_factor": _number(rows.get("minimum_safety_factor_max")),
        "max_unstable_fraction": _number(rows.get("max_unstable_fraction")),
        "max_flow_depth_m": _number(rows.get("maximum_flow_depth_m_max")),
        "max_velocity_ms": _number(rows.get("maximum_velocity_ms_max")),
        "max_pore_water_pressure_kpa": _number(rows.get("maximum_pore_water_pressure_kpa_max")),
        "max_unstable_duration_h": _number(rows.get("unstable_duration_h_max")),
        "peak_hourly_rain_mm": _number(rows.get("peak_hourly_rain_mm")),
        "rainfall_hours": _number(rows.get("rainfall_hours")),
        "mechanics_model": rows.get("mechanics_model"),
        "parameter_preset": rows.get("parameter_preset"),
        "swmm_runoff_source": rows.get("swmm_runoff_source"),
        "swmm_dynamic_hydrograph_used": rows.get("swmm_dynamic_hydrograph_used"),
        "swmm_subcatchments": _number(rows.get("swmm_subcatchments")),
        "metrics": rows,
    }
    raster = _first(project / 'rapid_safety_factor_results', '*_enhanced_minimum_safety_factor.tif')
    if raster is not None:
        from .result_distribution import safety_distribution
        try:
            result['spatial_distribution'] = safety_distribution(raster)
        except (OSError, ValueError, RuntimeError) as exc:
            result['spatial_distribution'] = {'available': False, 'reason': type(exc).__name__}
    return result


def _number(value: Any) -> float | None:
    result = _float(value)
    return None if result != result else round(result, 6)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rainfall_summary(project: Path) -> dict[str, Any]:
    station = _first(project / "inputs/rainfall", "*station_rainfall_summary.csv")
    active = _first(project / "inputs/rainfall", "*active_hourly.csv")
    result: dict[str, Any] = {"available": bool(station or active)}
    if station is not None:
        rows = _read_csv(station)
        result["station_count"] = len(rows)
        result["stations"] = [
            {key: row.get(key) for key in ("station_key", "station_name", "lon", "lat", "total_mm")
             if key in row}
            for row in rows[:10]
        ]
    if active is not None:
        rows = _read_csv(active)
        result["active_rows"] = len(rows)
        total = sum(_float(row.get("rain_mm"), 0.0) for row in rows)
        result["total_rain_mm"] = round(total, 4)
        peaks = [_float(row.get("rain_mm"), 0.0) for row in rows]
        result["peak_hourly_rain_mm"] = round(max(peaks), 4) if peaks else None
    return result


def summarize(job_dir: Path, project: Path) -> dict[str, Any]:
    # Snapshot of the imported package at submission, independent of later
    # session edits or spatial-service state. Older jobs remain readable.
    job = read_json(job_dir / "job.json") if (job_dir / "job.json").is_file() else {}
    artifacts = load_artifacts(job_dir)
    gates = read_gates(job_dir)
    by_stage: dict[str, int] = {}
    for record in artifacts:
        by_stage[record["stage_id"]] = by_stage.get(record["stage_id"], 0) + 1
    return {
        "artifact_count": len(artifacts),
        "artifacts_by_stage": by_stage,
        "swmm_model": inspect_swmm_model(project),
        "safety": safety_summary(project),
        "rainfall": rainfall_summary(project),
        "quality": None if gates is None else {
            "passed": gates.get("passed"),
            "failed": gates.get("failed", []),
            "checked_at": gates.get("checked_at"),
        },
        "domain_policy": job.get("domain_policy") or {},
        "limitations": list(dict.fromkeys([*(job.get("input_limitations") or []), ROUTING_LIMITATION])),
    }


def read_artifact_text(job_dir: Path, project: Path, artifact_id: str,
                       limit_bytes: int = 20 * 1024) -> dict[str, Any]:
    from .artifact_registry import load_artifacts as _load

    for record in _load(job_dir):
        if record["artifact_id"] != artifact_id:
            continue
        if Path(record["internal_path"]).suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError(f"成果不是文本类型，无法直接读取：{record['internal_path']}")
        from watershed_exchange.paths import safe_relative_path

        path = (project / safe_relative_path(record["internal_path"])).resolve()
        if project.resolve() not in path.parents and path != project.resolve():
            raise ValueError("成果路径超出运行目录")
        payload = read_text_limited(path, limit_bytes)
        payload.update({"artifact_id": artifact_id, "internal_path": record["internal_path"],
                        "media_type": record["media_type"]})
        return payload
    raise ValueError(f"未登记成果：{artifact_id}")
