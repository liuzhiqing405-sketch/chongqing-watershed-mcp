from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
import sys

import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT

CFG = LazyConfig()


def reset_dir(path: Path) -> None:
    if path.exists():
        try:
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        except PermissionError:
            stamped = path.with_name(f"{path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            stamped.mkdir(parents=True, exist_ok=True)
            print(f"[WARN] Existing final_results is locked; writing to: {stamped}")
            return stamped
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst_dir: Path, role: str, rows: list[dict]) -> None:
    if not src.exists() or not src.is_file():
        return
    if src.name.lower().endswith(".lock") or ".lock" in src.name.lower():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        try:
            shutil.copy2(src, dst)
        except PermissionError as exc:
            print(f"[WARN] Skip locked file: {src} ({exc})")
            return
    rows.append(
        {
            "role": role,
            "source": str(src),
            "final_file": str(dst),
            "size_mb": round(dst.stat().st_size / 1024 / 1024, 3),
        }
    )


def copy_folder_files(src_dir: Path, dst_dir: Path, role: str, rows: list[dict], suffixes: set[str] | None = None) -> None:
    if not src_dir.exists():
        return
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.name.lower().endswith(".lock") or ".lock" in src.name.lower():
            continue
        if "final_results" in [p.name for p in src.parents]:
            continue
        if suffixes and src.suffix.lower() not in suffixes:
            continue
        copy_file(src, dst_dir / src.relative_to(src_dir).parent, role, rows)


def main() -> None:
    final_dir = reset_dir(ROOT / "final_results")
    rows: list[dict] = []
    slug = CFG["project"]["name"]

    rain_dir = ROOT / "inputs" / "rainfall"
    copy_file(Path(CFG["rainfall"]["csv"]), final_dir / "01_rainfall", "rainfall_hourly_csv", rows)
    for key in ["station_csv", "station_observations_csv", "station_summary_csv", "station_domain_weights_csv"]:
        copy_file(Path(CFG["rainfall"].get(key) or ""), final_dir / "01_rainfall", "multi_station_rainfall", rows)
    copy_folder_files(rain_dir / "figures", final_dir / "01_rainfall", "rainfall_plot_and_stats", rows, {".png", ".pdf", ".csv"})

    copy_folder_files(ROOT / "inputs" / "dem_derived", final_dir / "02_dem_swmm_partition", "dem_derived_swmm_partition", rows)

    copy_file(Path(CFG["swmm"]["working_inp"]), final_dir / "03_swmm", "swmm_input", rows)
    copy_file(Path(CFG["swmm"]["report"]), final_dir / "03_swmm", "swmm_report", rows)
    copy_file(Path(CFG["swmm"]["output"]), final_dir / "03_swmm", "swmm_binary_output", rows)
    copy_file(ROOT / "swmm" / "model" / "auto_swmm_parameter_review.csv", final_dir / "03_swmm", "swmm_parameter_review", rows)
    copy_file(ROOT / "swmm" / "model" / "auto_swmm_rainfall_review.csv", final_dir / "03_swmm", "subcatchment_rainfall_review", rows)
    copy_file(ROOT / "swmm" / "model" / "auto_swmm_station_idw_weights.csv", final_dir / "03_swmm", "subcatchment_station_weights", rows)
    from ..swmm.landuse_response import FILES as RESPONSE_FILES
    for name in RESPONSE_FILES:
        copy_file(ROOT / "swmm/model" / name, final_dir / "03_swmm", "landuse_response_mapping", rows)
    copy_file(ROOT / "swmm/results/auto_swmm_parent_runoff.csv", final_dir / "03_swmm", "parent_runoff", rows)

    copy_folder_files(ROOT / "rapid_safety_factor_results", final_dir / "04_safety_factor", "safety_factor_result", rows)

    copy_folder_files(ROOT / "analysis_results", final_dir / "05_parameter_and_temporal_analysis", "parameter_and_temporal_analysis", rows)
    copy_file(ROOT / "最终运行与验收报告.md", final_dir, "validation_report", rows)

    manifest = final_dir / "final_results_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["role", "source", "final_file", "size_mb"])
        writer.writeheader()
        writer.writerows(rows)

    readme = final_dir / "README_最终结果说明.md"
    safety_rasters = [r for r in rows if r["role"] == "safety_factor_result" and r["final_file"].lower().endswith(".tif")]
    readme.write_text(
        "\n".join(
            [
                "# 最终结果文件夹说明",
                "",
                f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "本文件夹只保留安全系数主流程需要查看和交付的结果：降雨过程、DEM 自动分区、SWMM 模型文件，以及增强版安全系数栅格。",
                "",
                "## 目录结构",
                "",
                "- `01_rainfall`：10 个实测站逐时观测、站点统计、IDW 区域权重、面积平均过程和多站降雨图。",
                "- `02_dem_swmm_partition`：由 GRASS/DEM 自动提取的自然集水区、分区统计和预览图。",
                "- `03_swmm`：SWMM 输入文件、计算报告、二进制结果、子汇水区降雨复核表和十站 IDW 权重表。",
                "- `04_safety_factor`：主情景核心输出，包括最小安全系数、不稳定持续时间、最大水深、最大流速、最大孔隙水压力和土地利用力学参数分区。",
                "- `05_parameter_and_temporal_analysis`：区域与代表点单因素敏感性、Sobol 参数贡献及收敛性、分阶段稳定性和安全系数逐时变化图表。",
                "",
                "## 最重要的文件",
                "",
                "- `04_safety_factor/*_enhanced_minimum_safety_factor.tif`：最小安全系数栅格，Fs < 1 表示模拟达到不稳定状态。",
                "- `04_safety_factor/*_enhanced_unstable_duration_h.tif`：不稳定持续时间。",
                "- `04_safety_factor/*_enhanced_safety_factor_summary.csv`：安全系数统计摘要。",
                "- `05_parameter_and_temporal_analysis/parameter_sensitivity_oat.*`：单因素参数响应图。",
                "- `05_parameter_and_temporal_analysis/representative_point_sensitivity_oat.*`：R1、R2、N1 代表点的参数变化率与 Fs 变化率关系。",
                "- `05_parameter_and_temporal_analysis/parameter_contribution_sobol.*`：参数一阶和总效应贡献图。",
                "- `05_parameter_and_temporal_analysis/parameter_contribution_sobol_convergence.*`：不同基础样本量下的总效应指数收敛图。",
                "- `05_parameter_and_temporal_analysis/parameter_contribution_shap.*`：参数 SHAP 分布及平均绝对贡献图，并附代理模型精度。",
                "- `05_parameter_and_temporal_analysis/safety_factor_time_series.*`：降雨、代表点与区域安全系数及不稳定面积时序图。",
                "- `05_parameter_and_temporal_analysis/representative_point_fs_rainfall.*`：与参考论文对应的降雨柱和 R1/R2/N1 安全系数叠加图。",
                "- `05_parameter_and_temporal_analysis/staged_stability_comparison.*`：降雨前、峰值/临界时刻和计算结束时的稳定性等级面积组成。",
                "",
                "## 当前扫描结果",
                "",
                f"- 已收集文件数：{len(rows)}",
                f"- 安全系数相关 GeoTIFF 数：{len(safety_rasters)}",
                "",
                "所有文件来源和复制后的路径记录在 `final_results_manifest.csv`。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] Final results folder written: {final_dir}")
    print(f"[OK] Manifest: {manifest}")
    print(f"[OK] Readme: {readme}")
    print(f"[INFO] Files collected: {len(rows)}; safety rasters: {len(safety_rasters)}")


if __name__ == "__main__":
    main()
