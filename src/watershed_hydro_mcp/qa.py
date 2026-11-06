"""Quality gates for a hydro run.

Gates run as the final pipeline stage and can be re-run on a finished job. A
run whose computation succeeded but whose gates did not pass is reported as
``qa_failed`` — never as success.
"""

from __future__ import annotations

import csv
import re
import math
from pathlib import Path
from typing import Any

from .planner import load_gates
from .settings import SETTINGS, Settings
from .state import read_json, utc_now, write_json
from watershed_exchange.paths import sha256_file


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _first_file(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def spatial_gate(project: Path, gates: dict[str, Any]) -> dict[str, Any]:
    summary = _first_file(project / "inputs/dem_derived", "*subcatchments_summary.csv")
    rows = _read_csv_rows(summary) if summary else []
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {"subcatchment_count": len(rows)}
    diagnostic_path = project / "inputs/dem_derived/dem_derived_subcatchments_diagnostics.json"
    diagnostics = read_json(diagnostic_path) if diagnostic_path.is_file() else None
    checked_topology = bool(diagnostics and diagnostics.get("topology", {}).get("passed"))
    if not rows:
        errors.append("未生成DEM派生子汇水区摘要，无法核对空间划分")
    else:
        areas = [_float(row.get("area_ha")) for row in rows]
        if any(not math.isfinite(value) or value <= 0 for value in areas):
            errors.append("子汇水区摘要包含无效或非正面积")
        areas = [value for value in areas if value == value and value > 0]
        total = sum(areas)
        facts["total_area_ha"] = round(total, 4)
        if total <= 0:
            errors.append("子汇水区总面积为0")
        else:
            largest = max(areas)
            fraction = largest / total
            facts["largest_fraction"] = round(fraction, 4)
            limit = float(gates.get("maximum_single_cell_subcatchment_fraction", 0.35))
            if fraction > limit:
                (warnings if checked_topology else errors).append(
                    f"最大子汇水区占总面积 {fraction:.1%}，超过 {limit:.0%}，疑似划分未收敛"
                )
        minimum = float(gates.get("minimum_basin_area_km2", 1.8))
        if total * 0.01 < minimum:
            errors.append(f"研究区总面积 {total * 0.01:.3f} km2 小于下限 {minimum} km2")
        if any(row.get("source") in {"dem_delineation", "grass_r_watershed"} for row in rows) and not diagnostics:
            errors.append("DEM分区缺少新版填洼与排水拓扑验证，请重新划分；原空间QA不足以证明水文合理")
    if diagnostics:
        facts["drainage_diagnostics"] = diagnostics
        topology = diagnostics.get("topology") or {}
        version = diagnostics.get("schema_version")
        expected_algorithm = {1: "mask_boundary_priority_flood_d8_parent_flats", 2: "buffered_d8_local_partition_boundary_filter", 3: "grass_r_watershed_sfd_local_domain"}.get(version)
        if (expected_algorithm is None
                or diagnostics.get("algorithm") != expected_algorithm
                or not topology.get("passed")
                or topology.get("labelled_cells") != topology.get("valid_cells")
                or topology.get("labels_outside_domain") != 0
                or topology.get("internal_terminal_cells") != 0):
            errors.append("分区覆盖或排水拓扑验证未通过")
        if version == 2:
            filter_errors, filter_facts = boundary_filter_check(project, diagnostics)
            errors.extend(filter_errors)
            facts['boundary_filter'] = filter_facts
            if filter_facts.get('excluded_count', 0):
                warnings.append(f"边缘筛选排除 {filter_facts['excluded_count']} 个独立小分区、{filter_facts.get('excluded_area_ha', 0):.4f} ha；SWMM与稳定性采用保留范围")
            if not (diagnostics.get('drainage_context') or {}).get('used'):
                warnings.append('旧输入包没有外围DEM，本次仅使用选区内地形；重新准备输入可启用外围排水辅助')
        if not (diagnostics.get("model_budget") or {}).get("passed"):
            errors.append("分区数量超过模型计算预算")
        if version == 3:
            grass_errors, grass_facts = grass_partition_check(project, diagnostics, len(rows))
            errors.extend(grass_errors)
            facts["grass_partition"] = grass_facts
            coverage = grass_facts.get('retained_cells', 0) / max(1, grass_facts.get('original_cells', 0))
            if coverage < float(gates.get('minimum_coverage_fraction', .80)):
                errors.append('GRASS保留分区覆盖率低于空间质量门槛')
            if grass_facts.get("excluded_cells", 0):
                warnings.append(f"GRASS未分区或边缘筛选排除 {grass_facts['excluded_area_ha']:.4f} ha；计算采用明确导出的保留范围")
        else:
            coarsening = diagnostics.get("coarsening") or {}
            if coarsening.get("partition_count") != len(rows):
                errors.append("分区摘要数量与排水验证记录不一致")
            if not coarsening.get("major_target_reached"):
                errors.append("主要子汇水区数量未达到目标，请调整河网阈值或选择范围")
            if not coarsening.get("target_reached"):
                warnings.append("为保留独立的边界排水，细小边界单元单列；总分区数可多于主要分区目标，未跨分水岭强行合并")
            fraction = _float(diagnostics.get("fill_above_1m_fraction"))
            depth = _float(diagnostics.get("maximum_fill_depth_m"))
            if not math.isfinite(fraction) or not 0 <= fraction <= float(gates.get("maximum_fill_above_1m_fraction", .25)):
                errors.append("填洼大幅改变地形的区域过多，需要复核DEM与排水边界")
            if not math.isfinite(depth) or not 0 <= depth <= float(gates.get("maximum_fill_depth_m", 50)):
                errors.append("填洼最大抬高超过门槛，需要复核洼地与DEM")
        area = _float(diagnostics.get("valid_area_ha"))
        if not math.isfinite(area) or not math.isclose(area, facts.get("total_area_ha", 0), rel_tol=1e-6, abs_tol=1e-4):
            errors.append("矢量分区面积与原有效像元面积不一致")
        shape = _first_file(project / "inputs/dem_derived", "*subcatchments.shp")
        if (not shape or sha256_file(shape) != diagnostics.get("partition_sha256")
                or not summary or sha256_file(summary) != diagnostics.get("summary_sha256")):
            errors.append("分区文件与排水验证记录的校验和不一致，请重新划分")
        inp = _first_file(project / "swmm/model", "*.inp")
        if inp:
            model_text = inp.read_text(encoding="utf-8", errors="replace")
            section = re.search(r"^\s*\[SUBCATCHMENTS\]\s*$([\s\S]*?)(?=^\s*\[|\Z)", model_text, flags=re.M)
            model_rows = [line.split() for line in section.group(1).splitlines()
                          if line.strip() and not line.strip().startswith(";")] if section else []
            model_areas = [_float(row[3]) for row in model_rows if len(row) >= 4]
            facts["swmm_partition_count"] = len(model_rows)
            facts["swmm_total_area_ha"] = sum(model_areas)
            response_model = "; LANDUSE_REPRESENTATION response_units" in model_text
            response_valid = False
            if response_model:
                try:
                    from .core.swmm.landuse_response import validate_units
                    parent_parameters = _read_csv_rows(project / "swmm/model/auto_swmm_parameter_review.csv")
                    facts["landuse_response"] = validate_units(inp.parent, model_rows, parent_parameters)
                    response_valid = True
                    facts["swmm_partition_count"] = len(parent_parameters)
                    facts["swmm_response_unit_count"] = len(model_rows)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    errors.append(f"地类响应单元质量检查失败：{exc}")
            # Six decimal places in INP allow at most half a unit per row.
            tolerance = max(1e-5, len(model_rows) * 0.00000051)
            if ((not response_valid and (len(model_rows) != len(rows) or len(model_areas) != len(rows)))
                    or any(not math.isfinite(value) or value <= 0 for value in model_areas)
                    or abs(sum(model_areas) - area) > tolerance):
                errors.append("SWMM分区数量或总面积与已验证DEM分区不一致，可能重复拆分多面或把孔洞作为汇水区，请重新建模")
            parameter_path = project / "swmm/model/auto_swmm_parameter_review.csv"
            parameters = _read_csv_rows(parameter_path)
            names = [str(row.get("name") or "") for row in rows]
            sources = [str(row.get("source_name") or "") for row in parameters]
            if (len(parameters) != len(rows) or sorted(sources) != sorted(names)
                    or len(set(sources)) != len(sources)):
                errors.append("SWMM参数表未与水文分区一一对应，径流映射可能错配，请重新建模")
            if any(row.get('outlet_method') for row in parameters):
                outlet_errors, outlet_facts = model_outlet_check(project, shape, parameters, model_text)
                errors.extend(outlet_errors)
                facts['model_outlets'] = outlet_facts
    return {"name": "spatial", "passed": not errors, "errors": errors, "warnings": warnings, "facts": facts}


def grass_partition_check(project: Path, diagnostics: dict, count: int) -> tuple[list[str], dict]:
    """GRASS has no filled DEM: validate its routing and excluded domain directly."""
    import numpy as np
    import rasterio
    from .core.delineation.grass_delineation import drainage_receivers
    from .core.delineation.delineate import accumulation, domain_boundary, outlet_points_by_label
    from .core.delineation.partition_topology import partition_facts, single_exit_facts
    errors, facts = boundary_filter_check(project, diagnostics)
    grass = diagnostics.get('grass') or {}
    partitioning = diagnostics.get('partitioning') or {}
    if partitioning.get('partition_count') != count or partitioning.get('coarsened') is not False:
        errors.append('GRASS分区数量不一致或原始分区被意外合并')
    root = project / 'inputs/dem_derived'
    expected = {'basin.tif', 'drainage.tif', 'accumulation.tif', 'stream.tif'}
    if set(grass.get('rasters') or {}) != expected or grass.get('flags') != 's':
        errors.append('GRASS原始栅格或SFD计算参数记录缺失')
    for name in expected:
        path = root/'grass'/name
        if not path.is_file() or sha256_file(path) != (grass.get('rasters') or {}).get(name):
            errors.append(f'GRASS原始结果缺失或校验和不一致：{name}')
    dem_path = root/'grass/dem_valid.tif'
    if not dem_path.is_file() or sha256_file(dem_path) != grass.get('dem_sha256'):
        errors.append('GRASS计算DEM缺失或校验和不一致')
    source_dem = project/'inputs/user_data/dem.tif'
    if not source_dem.is_file() or sha256_file(source_dem) != grass.get('source_dem_sha256'):
        errors.append('GRASS原始输入DEM与划分记录不一致')
    if errors:
        return errors, facts
    with rasterio.open(root/'calculation_mask.tif') as src:
        retained, transform, crs = src.read(1).astype(bool), src.transform, src.crs
    arrays = {}
    for name in ('basin', 'drainage'):
        with rasterio.open(root/f'grass/{name}.tif') as src:
            if src.shape != retained.shape or src.crs != crs or not src.transform.almost_equals(transform):
                errors.append('GRASS栅格与计算范围不对齐')
                return errors, facts
            arrays[name] = src.read(1)
    with rasterio.open(dem_path) as src:
        elevations = src.read(1, masked=True)
        original = ~np.ma.getmaskarray(elevations)
        elevations = elevations.filled(np.nan)
    import yaml
    config = yaml.safe_load((project/'config.yaml').read_text(encoding='utf-8'))
    threshold_ha = float(config['swmm'].get('channel_threshold_ha', 20))
    threshold_cells = max(1, round(threshold_ha/(abs(transform.a*transform.e)/10000)))
    if (grass.get('threshold_cells') != threshold_cells
            or not math.isclose(_float(grass.get('threshold_ha')), threshold_ha, abs_tol=1e-9)):
        errors.append('GRASS河网阈值与本次运行配置不一致')
    unassigned = original & (arrays['basin'] == 0)
    fraction = unassigned.sum()/max(1, original.sum())
    if (grass.get('unassigned_cells') != int(unassigned.sum())
            or not math.isclose(_float(grass.get('unassigned_fraction')), fraction, abs_tol=1e-9)
            or not 0 <= fraction <= _float(grass.get('maximum_unassigned_fraction'))):
        errors.append('GRASS未分区面积记录不一致或超过容差')
    labels = np.where(retained, arrays['basin'], 0).astype('int32')
    receivers = drainage_receivers(arrays['drainage'], retained)
    ids = np.flatnonzero(retained.ravel() & (receivers >= 0))
    steps = elevations.ravel()[receivers[ids]]-elevations.ravel()[ids]
    facts['least_cost_routing'] = {
        'uphill_step_cells': int(np.count_nonzero(steps > 1e-6)),
        'maximum_uphill_step_m': float(max(0, steps.max())) if len(steps) else 0,
        'interpretation': 'GRASS least-cost routing may cross depressions; original DEM elevations are unchanged',
    }
    try:
        acc = accumulation(receivers, retained)
    except RuntimeError:
        return errors + ['GRASS排水流向存在环路'], facts
    topology = partition_facts(labels, receivers, retained, domain_boundary(retained))
    if not topology['passed'] or topology != diagnostics.get('topology'):
        errors.append('GRASS排水拓扑与复核结果不一致')
    single_exit = single_exit_facts(labels, receivers, retained)
    facts['single_exit'] = single_exit
    if not single_exit['passed']:
        errors.append(f"GRASS分区未汇向唯一出口：{single_exit['invalid_partition_count']} 个分区缺少出口或存在多个出流位置")
    if 'single_exit' in partitioning and single_exit != partitioning['single_exit']:
        errors.append('GRASS唯一出口审计与原始流向复核结果不一致')
    outlets = outlet_points_by_label(labels, receivers, acc, transform)
    summary = _read_csv_rows(root/'dem_derived_subcatchments_summary.csv')
    import shapefile
    shape = _first_file(root, '*subcatchments.shp')
    with shapefile.Reader(str(shape)) as reader:
        records = {r.as_dict()['name']: r.as_dict() for r in reader.records()}
    from .core.spatial_domain import shapefile_polygon_features
    from rasterio.features import rasterize
    label_by_name = {r['name']: int(r['label']) for r in summary}
    features = shapefile_polygon_features(shape, crs)
    painted = rasterize([(f['geometry'], label_by_name.get(f['record'].get('name'), 0))
                         for f in features], out_shape=labels.shape, transform=transform,
                        fill=0, dtype='int32')
    if not np.array_equal(painted, labels):
        errors.append('矢量分区与GRASS原始分区标识不一致')
    for row in summary:
        label = int(row['label'])
        record = records.get(row['name']) or {}
        expected_point = outlets.get(label)
        if expected_point is None or any(abs(_float(record.get(key))-value) > .001
                                        for key, value in zip(('out_x', 'out_y'), expected_point)):
            errors.append('GRASS分区出流点与原始流向不一致')
            break
    facts['partition_count'] = count
    facts['unassigned_area_ha'] = float(unassigned.sum()*abs(transform.a*transform.e)/10000)
    return errors, facts


def boundary_filter_check(project: Path, diagnostics: dict) -> tuple[list[str], dict]:
    import numpy as np
    import rasterio
    errors = []
    facts = diagnostics.get('boundary_filter') or {}
    root = project / 'inputs/dem_derived'
    files = {'calculation_mask.tif': 'calculation_mask_sha256', 'excluded_boundary_mask.tif': 'excluded_boundary_mask_sha256',
             'excluded/boundary_fragments.shp': 'excluded_shape_sha256',
             'excluded/boundary_fragments_summary.csv': 'excluded_summary_sha256'}
    for name, key in files.items():
        path = root/name
        if not path.is_file() or sha256_file(path) != facts.get(key):
            errors.append(f'边缘筛选审计文件缺失或校验和不一致：{name}')
    if errors:
        return errors, facts
    with rasterio.open(root/'calculation_mask.tif') as a, rasterio.open(root/'excluded_boundary_mask.tif') as b:
        retained, excluded = a.read(1), b.read(1)
        if a.shape != b.shape or a.transform != b.transform or a.crs != b.crs:
            return ['保留与排除范围栅格不对齐'], facts
        if not np.isin(retained, [0, 1]).all() or not np.isin(excluded, [0, 1]).all() or np.any(retained & excluded):
            errors.append('保留与排除范围存在重叠或无效像元')
        cell_area = abs(a.transform.a*a.transform.e)/10000
        n, m = int(retained.sum()), int(excluded.sum())
        from rasterio.features import rasterize
        from rasterio.warp import reproject, Resampling
        from .core.spatial_domain import shapefile_polygon_features, landuse_mask_for_grid
        shape = _first_file(root, '*subcatchments.shp')
        for path, mask in [(shape, retained), (root/'excluded/boundary_fragments.shp', excluded)]:
            features = shapefile_polygon_features(path, a.crs)
            painted = rasterize([(f['geometry'], 1) for f in features], out_shape=a.shape,
                                transform=a.transform, fill=0, dtype='uint8') if features else np.zeros(a.shape, dtype='uint8')
            if not np.array_equal(painted, mask):
                errors.append('筛选分区几何与计算范围栅格不一致')
        with rasterio.open(project/'inputs/user_data/dem.tif') as source:
            data = source.read(1, masked=True).astype('float64').filled(np.nan)
            original = np.full(a.shape, np.nan)
            reproject(data, original, src_transform=source.transform, src_crs=source.crs,
                      dst_transform=a.transform, dst_crs=a.crs, src_nodata=np.nan, dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        original_valid = np.isfinite(original)
        import yaml
        config = yaml.safe_load((project/'config.yaml').read_text(encoding='utf-8'))
        if (config.get('study_area') or {}).get('calculation_domain') == 'dem_landuse_intersection':
            original_valid &= landuse_mask_for_grid(a.shape, a.transform, a.crs, project/'inputs/user_data/landuse.shp')
        if not np.array_equal(original_valid, (retained | excluded).astype(bool)):
            errors.append('保留和排除范围未完整覆盖原选区有效像元')
    context = diagnostics.get('drainage_context') or {}
    if context.get('used'):
        path = project/'inputs/user_data/dem_context.tif'
        if not path.is_file() or sha256_file(path) != context.get('sha256'):
            errors.append('外围排水DEM与验证记录不一致')
    for key, value in [('retained_cells', n), ('excluded_cells', m), ('original_cells', n+m)]:
        if facts.get(key) != value:
            errors.append(f'边缘筛选面积审计不一致：{key}')
    for key, value in [('retained_area_ha', n*cell_area), ('excluded_area_ha', m*cell_area), ('original_area_ha', (n+m)*cell_area)]:
        if not math.isclose(_float(facts.get(key)), value, abs_tol=1e-5):
            errors.append(f'边缘筛选面积审计不一致：{key}')
    if not math.isclose(_float(diagnostics.get('valid_area_ha')), n*cell_area, abs_tol=1e-5):
        errors.append('模型面积不等于筛选后保留面积')
    fraction = m/max(1, n+m)
    if (not math.isclose(_float(facts.get('excluded_fraction')), fraction, abs_tol=1e-9)
            or fraction > _float(facts.get('maximum_excluded_fraction')) + 1e-12):
        errors.append('边缘排除面积超过已声明容差或比例记录不一致')
    rows = _read_csv_rows(root/'excluded/boundary_fragments_summary.csv')
    if len(rows) != facts.get('excluded_count') or not math.isclose(sum(_float(r.get('area_ha')) for r in rows), m*cell_area, abs_tol=1e-4):
        errors.append('排除分区清单与排除面积不一致')
    return errors, facts


def model_outlet_check(project: Path, partitions: Path, parameters: list[dict], inp_text: str) -> tuple[list[str], dict]:
    """Verify the chosen outlet against native DEM, full GIS and rounded INP."""
    import rasterio
    from pyproj import CRS
    from .core.spatial_domain import shapefile_polygon_features
    from .core.swmm.build_inp import minimum_dem_outlet, drainage_exit_outlet

    def section_rows(name):
        match = re.search(rf"^\s*\[{name}\]\s*$([\s\S]*?)(?=^\s*\[|\Z)", inp_text, flags=re.M)
        return [line.split() for line in match.group(1).splitlines()
                if line.strip() and not line.strip().startswith(';')] if match else []

    coordinates = {row[0]: row[1:3] for row in section_rows('COORDINATES') if len(row) >= 3}
    elevations = {row[0]: row[1] for row in section_rows('OUTFALLS') if len(row) >= 2}
    model_outlets = {row[0]: row[2] for row in section_rows('SUBCATCHMENTS') if len(row) >= 3}
    from .core.swmm.landuse_response import read_units
    response_outlets = {}
    for unit in read_units(project / 'swmm/model'):
        response_outlets.setdefault(unit['parent_subcatchment'], []).append(model_outlets.get(unit['subcatchment']))
    errors = []
    checked = 0
    methods = {}
    with rasterio.open(project / 'inputs/user_data/dem.tif') as dem:
        features = shapefile_polygon_features(partitions, CRS.from_user_input(dem.crs))
        by_name = {str(item['record'].get('name') or item['record'].get('NAME') or ''): item for item in features}
        for row in parameters:
            source_name = str(row.get('source_name') or '')
            outlet = str(row.get('outfall') or '')
            method = row.get('outlet_method')
            if method not in {'minimum_dem_elevation', 'drainage_exit'} or source_name not in by_name:
                errors.append(f"{source_name} 的出口规则或来源几何缺失")
                continue
            item = by_name[source_name]
            expected = (drainage_exit_outlet(item['record'], item['geometry'], dem)
                        if method == 'drainage_exit' else minimum_dem_outlet(item['geometry'], dem))
            if expected is None:
                errors.append(f'{source_name} 缺少沿流向确定的排水出口')
                continue
            methods[method] = methods.get(method, 0) + 1
            x, y = expected['outlet']
            values = coordinates.get(outlet, [None, None])
            if (not math.isclose(_float(row.get('outlet_x')), x, abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(_float(row.get('outlet_y')), y, abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(_float(row.get('outfall_elev_m')), expected['min_elev'], abs_tol=1e-6, rel_tol=0)
                    or not math.isclose(_float(values[0]), x, abs_tol=.00051, rel_tol=0)
                    or not math.isclose(_float(values[1]), y, abs_tol=.00051, rel_tol=0)
                    or not math.isclose(_float(elevations.get(outlet)), expected['min_elev'], abs_tol=.00051, rel_tol=0)
                    or (any(value != outlet for value in response_outlets[str(row.get('subcatchment'))])
                        if str(row.get('subcatchment')) in response_outlets
                        else model_outlets.get(str(row.get('subcatchment'))) != outlet)
                    or _float(row.get('outlet_dem_row')) != expected['outlet_dem_row']
                    or _float(row.get('outlet_dem_col')) != expected['outlet_dem_col']):
                errors.append(f"{source_name} 的出口坐标、高程或模型连接不符合所记录的出口规则")
            else:
                checked += 1
    return errors[:10], {'methods':methods, 'verified_count':checked, 'expected_count':len(parameters)}


def rainfall_gate(project: Path, gates: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    facts: dict[str, Any] = {}
    station_summary = _first_file(project / "inputs/rainfall", "*station_rainfall_summary.csv")
    observations = _first_file(project / "inputs/rainfall", "*station_observations_long.csv")
    active = _first_file(project / "inputs/rainfall", "*active_hourly.csv")
    if not station_summary:
        errors.append("未生成站点降雨汇总表")
    else:
        rows = _read_csv_rows(station_summary)
        facts["stations"] = len(rows)
        if len(rows) == 1:
            facts["spatial_rainfall_assumption"] = "single_station_uniform"
        if len(rows) < int(gates.get("minimum_stations", 1)):
            errors.append(f"雨站 {len(rows)} 个，少于下限 {gates.get('minimum_stations')}")
        peaks = [_float(row.get("peak_mm_h"), 0.0) for row in rows]
        peaks = [value for value in peaks if value == value]
        if peaks:
            facts["max_station_hourly_mm"] = round(max(peaks), 4)
            limit = float(gates.get("maximum_hourly_rain_mm", 200.0))
            if max(peaks) > limit:
                errors.append(f"站点最大小时雨量 {max(peaks):.1f} mm 超过 {limit} mm，疑似单位或录入错误")
        totals = [_float(row.get("total_mm"), 0.0) for row in rows]
        if totals and max(totals) <= 0:
            errors.append("全部站点降雨总量为0，事件窗口可能选错")
    if observations:
        rows = _read_csv_rows(observations)
        timestamps = {row.get("datetime_beijing") for row in rows}
        facts["timestamps"] = len(timestamps)
        if len(timestamps) < int(gates.get("minimum_timestamps", 2)):
            errors.append(f"逐时记录 {len(timestamps)} 条，少于下限 {gates.get('minimum_timestamps')}")
        limit = int(gates.get("maximum_timestamps", 247))
        if len(timestamps) > limit:
            errors.append(f"逐时记录 {len(timestamps)} 条，超过雨站格式上限 {limit}")
    else:
        errors.append("未生成站点逐时降雨长表")
    if not active:
        errors.append("未生成流域面平均逐时降雨")
    else:
        rows = _read_csv_rows(active)
        total = sum(_float(row.get("rain_mm"), 0.0) for row in rows)
        facts["total_rain_mm"] = round(total, 4)
        if total < float(gates.get("minimum_total_rain_mm", 0.1)):
            errors.append(f"事件总雨量 {total:.3f} mm 接近0，事件窗口可能选错")
    return {"name": "rainfall", "passed": not errors, "errors": errors, "facts": facts}


def _report_table_rows(report_text: str, heading: str) -> list[list[str]]:
    """Data rows of a SWMM summary table, keyed by its heading.

    SWMM pads these tables with banner lines, multi-line column headers and
    dashed rules, so a row counts only when its first field is an identifier
    and at least two of the remaining fields are numbers. Collection stops once
    the table has produced rows and the data run ends.
    """
    markers = list(re.finditer(rf"^\s*{re.escape(heading)}\s*$", report_text, flags=re.M))
    if not markers:
        return []
    # The summary tables repeat the heading; the last one holds the actual run.
    tail = report_text[markers[-1].end():]
    rows: list[list[str]] = []
    misses = 0
    for line in tail.splitlines():
        stripped = line.strip()
        is_rule = stripped and (set(stripped) <= {"-"} or set(stripped) <= {"*"})
        if not stripped or is_rule:
            misses += 1
            if rows and misses >= 2:
                break
            continue
        parts = stripped.split()
        numeric = [_float(token) for token in parts[1:]] if len(parts) > 1 else []
        matches = sum(1 for value in numeric if value == value)
        # A data row is an identifier followed almost entirely by numbers; the
        # free-text lines SWMM appends after the table do not qualify.
        if len(parts) >= 3 and matches >= 3 and matches >= 0.6 * len(numeric):
            rows.append(parts)
            misses = 0
        else:
            misses += 1
            if rows and misses >= 2:
                break
    return rows


def swmm_gate(project: Path, gates: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}
    report = _first_file(project / "swmm/results", "*.rpt")
    if not report:
        return {"name": "swmm", "passed": False, "errors": ["未找到SWMM报告文件"], "facts": {}}
    text = report.read_text(encoding="utf-8", errors="replace")
    facts["report"] = report.name
    facts["error_lines"] = [
        line.strip() for line in text.splitlines()
        if line.upper().lstrip().startswith(("ERROR ", "ERROR:", "*** ERROR"))
    ][:10]
    if facts["error_lines"]:
        errors.append("SWMM报告包含错误行：" + facts["error_lines"][0])
    warnings_found = [line.strip() for line in text.splitlines() if "WARNING" in line.upper()]
    facts["warnings"] = warnings_found[:10]
    if warnings_found:
        warnings.append(f"SWMM报告含 {len(warnings_found)} 条警告，需人工复核")
    for label, key, limit_key, block in (
        ("径流连续性误差", "runoff_continuity_error_percent",
         "maximum_continuity_error_percent", "Runoff Quantity Continuity"),
        ("流量演算连续性误差", "flow_routing_continuity_error_percent",
         "maximum_flow_routing_error_percent", "Flow Routing Continuity"),
    ):
        value = float("nan")
        match = re.search(
            rf"{re.escape(block)}[\s\S]{{0,800}}?Continuity Error \(%\)\s*\.*\s*(-?\d+\.?\d*)", text)
        if match:
            value = _float(match.group(1))
        facts[key] = None if value != value else round(value, 4)
        if value != value:
            warnings.append(f"报告中未找到{label}，该阶段可能未执行流量演算")
            continue
        limit = float(gates.get(limit_key, 10.0))
        if abs(value) > limit:
            errors.append(f"{label} {value:.2f}% 超过 {limit}% 上限")
    subcatchment_rows = _report_table_rows(text, "Subcatchment Runoff Summary")
    facts["subcatchment_rows"] = len(subcatchment_rows)
    minimum = int(gates.get("minimum_subcatchments", 1))
    if len(subcatchment_rows) < minimum:
        errors.append(f"报告列出 {len(subcatchment_rows)} 个子汇水区，少于下限 {minimum}")
    if gates.get("require_outfalls", True):
        inp = _first_file(project / "swmm/model", "*.inp")
        outfall_count = 0
        if inp is not None:
            inp_text = inp.read_text(encoding="utf-8", errors="replace")
            section = re.search(r"^\s*\[OUTFALLS\]\s*$([\s\S]*?)(?=^\s*\[|\Z)", inp_text, flags=re.M)
            if section:
                outfall_count = len([
                    line for line in section.group(1).splitlines()
                    if line.strip() and not line.strip().startswith(";")
                ])
        facts["outfall_count"] = outfall_count
        if outfall_count < 1:
            errors.append("模型中没有任何出水口")
    return {"name": "swmm", "passed": not errors, "errors": errors,
            "warnings": warnings, "facts": facts}


def stability_gate(project: Path, gates: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}
    summary = _first_file(project / "rapid_safety_factor_results", "*_safety_factor_summary.csv")
    if not summary:
        return {"name": "stability", "passed": False,
                "errors": ["未找到安全系数汇总文件"], "facts": {}}
    rows = {row.get("metric"): row.get("value") for row in _read_csv_rows(summary)}
    facts["mechanism_model"] = rows.get("mechanics_model")
    facts["parameter_preset"] = rows.get("parameter_preset")
    minimum = _float(rows.get("minimum_safety_factor_min"))
    facts["minimum_safety_factor"] = None if minimum != minimum else round(minimum, 4)
    maximum = _float(rows.get("minimum_safety_factor_max"))
    facts["maximum_safety_factor"] = None if maximum != maximum else round(maximum, 4)
    mean_value = _float(rows.get("minimum_safety_factor_mean"))
    facts["mean_safety_factor"] = None if mean_value != mean_value else round(mean_value, 4)
    if minimum != minimum:
        errors.append("安全系数汇总缺少全局最小值")
    elif minimum <= 0:
        errors.append(f"最小安全系数为 {minimum:.3f}，非正值表示结果不可用")
    ceiling = float(gates.get("maximum_safety_factor", 100.0))
    if maximum == maximum and maximum > ceiling:
        warnings.append(f"最大安全系数 {maximum:.1f} 超过 {ceiling}，部分像元可能未进入失稳状态")
    unstable = _float(rows.get("max_unstable_fraction"))
    facts["max_unstable_fraction"] = None if unstable != unstable else round(unstable, 6)
    limit = float(gates.get("maximum_unstable_fraction", 0.95))
    if unstable == unstable and unstable > limit:
        warnings.append(
            f"最大失稳面积比例 {unstable:.1%} 超过 {limit:.0%}，需检查参数与边界条件"
        )
    sample = _float(rows.get("analysis_sample_cells"))
    facts["analysis_sample_cells"] = None if sample != sample else int(sample)
    if not rows.get("swmm_dynamic_source"):
        warnings.append("未记录SWMM动态径流来源，无法确认水力过程来自模拟结果")
    if str(rows.get("swmm_dynamic_hydrograph_used", "")).lower() not in {"true", "1"}:
        warnings.append("稳定性计算未使用SWMM动态径流过程，结果为降雨驱动近似")
    mask_path = project/'inputs/dem_derived/calculation_mask.tif'
    diagnostic_path = mask_path.with_name('dem_derived_subcatchments_diagnostics.json')
    if diagnostic_path.is_file() and read_json(diagnostic_path).get('schema_version') in {2, 3}:
        result_path = _first_file(project/'rapid_safety_factor_results', '*enhanced_minimum_safety_factor.tif')
        if not result_path or not mask_path.is_file():
            errors.append('筛选后的稳定性结果或计算范围缺失')
        else:
            import numpy as np
            import rasterio
            from .core.spatial_domain import apply_calculation_mask
            with rasterio.open(result_path) as source:
                values = source.read(1, masked=True)
                actual = ~np.ma.getmaskarray(values) & np.isfinite(np.ma.getdata(values))
                expected = apply_calculation_mask(np.ones(source.shape, bool), source.transform, source.crs, project)
                if not np.array_equal(actual, expected):
                    errors.append('稳定性有效计算范围与SWMM保留分区不一致')
                facts['retained_calculation_cells'] = int(expected.sum())
    return {"name": "stability", "passed": not errors, "errors": errors,
            "warnings": warnings, "facts": facts}


def export_gate(project: Path, gates: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    facts: dict[str, Any] = {}
    manifest = _first_file(project / "final_results", "final_results_manifest.csv")
    rows = _read_csv_rows(manifest) if manifest else []
    facts["artifact_count"] = len(rows)
    facts["manifest"] = "final_results/final_results_manifest.csv"
    if not rows:
        errors.append("未生成最终成果清单")
    else:
        minimum = int(gates.get("minimum_artifacts", 5))
        if len(rows) < minimum:
            errors.append(f"最终成果只有 {len(rows)} 项，少于下限 {minimum}")
        missing = []
        for row in rows:
            final_file = row.get("final_file")
            if not final_file:
                continue
            try:
                candidate = Path(final_file)
            except (TypeError, ValueError):
                continue
            if not candidate.is_absolute():
                candidate = project / candidate
            if not candidate.exists():
                missing.append(final_file)
        if missing:
            errors.append(f"清单中有 {len(missing)} 项成果在磁盘上不存在")
        facts["missing_count"] = len(missing)
        facts["roles"] = sorted({row.get("role") or "" for row in rows} - {""})
    return {"name": "export", "passed": not errors, "errors": errors, "facts": facts}


GATES = (
    ("spatial", spatial_gate),
    ("rainfall", rainfall_gate),
    ("swmm", swmm_gate),
    ("stability", stability_gate),
    ("export", export_gate),
)

#: Which gates apply once a given stage has completed.
STAGE_GATES = {
    "delineate": ("spatial",),
    "rainfall": ("rainfall",),
    "swmm": ("swmm",),
    "safety": ("stability",),
    "export": ("export",),
}


def compact_report(report):
    """Keep tool replies small; the full per-unit audit stays in qa.json."""
    if isinstance(report, dict):
        result = {}
        for key, value in report.items():
            if key == "excluded_units" and isinstance(value, list) and len(value) > 20:
                result[key] = value[:20]
                result["excluded_units_total"] = len(value)
                result["excluded_units_truncated"] = True
                result["excluded_units_full_record"] = "qa.json"
            else:
                result[key] = compact_report(value)
        return result
    if isinstance(report, list):
        return [compact_report(value) for value in report]
    return report


def run_gates(
    project: Path,
    completed_stages: list[str],
    settings: Settings = SETTINGS,
) -> dict[str, Any]:
    """Evaluate every gate whose stage actually ran."""
    gates = load_gates(settings)
    wanted: list[str] = []
    for stage in completed_stages:
        for name in STAGE_GATES.get(stage, ()):
            if name not in wanted:
                wanted.append(name)
    if "delineate" in completed_stages and "swmm" in completed_stages:
        # The full chain ran; require every gate that has inputs on disk.
        wanted = [name for name, _ in GATES]
    if not wanted:
        # A preflight run produced nothing a gate could judge; saying "passed"
        # with no gates is honest only if it is spelled out.
        return {
            "checked_at": utc_now(),
            "passed": True,
            "gates": [],
            "failed": [],
            "gates_applied": False,
            "note": "本次运行未包含需要质量检查的阶段，未执行任何门槛",
            "completed_stages": completed_stages,
        }
    results = []
    for name, function in GATES:
        if name not in wanted:
            continue
        try:
            results.append(function(project, gates.get(name) or {}))
        except Exception as exc:  # noqa: BLE001 - a gate must never crash a run
            results.append({"name": name, "passed": False,
                            "errors": [f"质量检查执行失败：{exc}"], "facts": {}})
    failed = [item for item in results if not item["passed"]]
    return {
        "checked_at": utc_now(),
        "passed": not failed,
        "gates": results,
        "failed": [item["name"] for item in failed],
        "gates_applied": True,
        "completed_stages": completed_stages,
    }


def write_gates(job_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    write_json(job_dir / "qa.json", report)
    return report


def read_gates(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / "qa.json"
    if not path.is_file():
        return None
    return read_json(path)
