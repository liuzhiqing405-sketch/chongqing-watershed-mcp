from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT

CFG = LazyConfig()


def slug() -> str:
    return CFG["project"]["name"]


def source_dir() -> Path:
    return ROOT / "rapid_safety_factor_results"


def out_dir() -> Path:
    return ROOT / "analysis_results"


PARAMETERS = [
    ("natural_density_g_cm3", "Natural density", "g cm$^{-3}$"),
    ("saturated_density_g_cm3", "Saturated density", "g cm$^{-3}$"),
    ("friction_angle_deg", "Friction angle", "deg"),
    ("cohesion_kpa", "Cohesion", "kPa"),
    ("runoff_density_g_cm3", "Runoff density", "g cm$^{-3}$"),
]

DEFAULT_RANGES = {
    "natural_density_g_cm3": [1.8, 2.0],
    "saturated_density_g_cm3": [1.9, 2.1],
    "friction_angle_deg": [15.0, 25.0],
    "cohesion_kpa": [5.0, 25.0],
    "runoff_density_g_cm3": [0.5, 1.5],
}

COLORS = {
    "blue": "#2F6EA6",
    "red": "#C94C4C",
    "gold": "#D49A32",
    "green": "#3D8062",
    "grey": "#727B84",
    "light": "#D9DEE3",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in [
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
    ]:
        fig.savefig(out_dir() / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Required input is missing: {path}. Run the safety-factor stage first.")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_state() -> dict[str, np.ndarray | str]:
    path = source_dir() / f"{slug()}_critical_state_parameter_sample.csv"
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"Critical-state sample is empty: {path}")
    numeric_fields = [
        "flat_index",
        "slope_angle_deg",
        "flow_depth_m",
        "flow_velocity_ms",
        "friction_velocity_ms",
        "pore_pressure_kpa",
        "initiation_shear_kpa",
        "resisting_shear_kpa",
        "cohesion_kpa",
        "friction_angle_deg",
        "soil_thickness_m",
        "natural_density_g_cm3",
        "saturated_density_g_cm3",
        "runoff_density_g_cm3",
        "base_fs",
    ]
    state: dict[str, np.ndarray | str] = {
        "critical_time": rows[0]["critical_time"],
        "representative_code": np.asarray([row.get("representative_code", "") for row in rows]),
    }
    for field in numeric_fields:
        state[field] = np.asarray([float(row[field]) for row in rows], dtype="float64")
    valid = np.ones(len(rows), dtype=bool)
    for field in numeric_fields:
        valid &= np.isfinite(np.asarray(state[field]))
    for field in numeric_fields:
        state[field] = np.asarray(state[field])[valid]
    state["representative_code"] = np.asarray(state["representative_code"])[valid]
    if int(np.count_nonzero(valid)) < 1000:
        raise RuntimeError("Too few valid critical-state sample cells for parameter analysis.")
    return state


def analysis_settings(state: dict[str, np.ndarray | str]) -> tuple[dict[str, tuple[float, float]], np.ndarray, int, int, int]:
    cfg = CFG.get("analysis", {})
    ranges_cfg = cfg.get("parameter_ranges", {})
    ranges: dict[str, tuple[float, float]] = {}
    for key, _, _ in PARAMETERS:
        raw = ranges_cfg.get(key, DEFAULT_RANGES[key])
        lo, hi = float(raw[0]), float(raw[1])
        if not lo < hi:
            raise ValueError(f"Invalid parameter range for {key}: {raw}")
        ranges[key] = (lo, hi)

    st_cfg = CFG.get("stability", {})
    baseline_cohesion = float(np.nanmean(np.asarray(state["cohesion_kpa"], dtype="float64")))
    baseline_phi = float(np.nanmean(np.asarray(state["friction_angle_deg"], dtype="float64")))
    baseline = np.asarray(
        [
            float(st_cfg.get("natural_density_g_cm3", 1.85)),
            float(st_cfg.get("saturated_density_g_cm3", 2.1)),
            baseline_phi,
            baseline_cohesion,
            float(st_cfg.get("runoff_density_g_cm3", 1.5)),
        ],
        dtype="float64",
    )
    levels = max(5, int(cfg.get("oat_levels", 7)))
    sobol_n = max(64, int(cfg.get("sobol_base_samples", 2048)))
    sobol_n = 2 ** int(math.ceil(math.log2(sobol_n)))
    bootstrap = max(50, int(cfg.get("sobol_bootstrap", 200)))
    return ranges, baseline, levels, sobol_n, bootstrap


def evaluate(samples: np.ndarray, state: dict[str, np.ndarray | str], batch_size: int = 32) -> dict[str, np.ndarray]:
    slope = np.deg2rad(np.asarray(state["slope_angle_deg"], dtype="float64"))
    sin_t = np.sin(slope)[None, :]
    tan_t = np.tan(slope)[None, :]
    velocity = np.asarray(state["flow_velocity_ms"], dtype="float64")[None, :]
    cohesion0 = np.asarray(state["cohesion_kpa"], dtype="float64")[None, :]
    phi0 = np.asarray(state["friction_angle_deg"], dtype="float64")[None, :]
    h0 = np.asarray(state["soil_thickness_m"], dtype="float64")[None, :]
    baseline_cohesion = max(float(np.nanmean(cohesion0)), 1e-6)
    baseline_phi = float(np.nanmean(phi0))
    rho_w = float(CFG.get("stability", {}).get("water_density_g_cm3", 1.0))
    friction_a = float(CFG.get("stability", {}).get("friction_velocity_coefficient_a", 0.6))
    friction_velocity = -0.44 * (1.0 - 1.0 / friction_a) * velocity

    mean_fs = np.empty(samples.shape[0], dtype="float64")
    unstable_fraction = np.empty(samples.shape[0], dtype="float64")
    representative_codes = np.asarray(state.get("representative_code", []))
    representative_positions = {
        code: int(np.flatnonzero(representative_codes == code)[0])
        for code in ["R1", "R2", "N1"]
        if np.any(representative_codes == code)
    }
    representative_fs = {
        code: np.empty(samples.shape[0], dtype="float64") for code in representative_positions
    }
    for start in range(0, samples.shape[0], batch_size):
        stop = min(start + batch_size, samples.shape[0])
        p = samples[start:stop]
        rho_s = p[:, 0, None]
        rho_sat = p[:, 1, None]
        phi = np.deg2rad(np.clip(phi0 + (p[:, 2, None] - baseline_phi), 1.0, 45.0))
        cohesion = cohesion0 * (p[:, 3, None] / baseline_cohesion)
        rho_runoff = p[:, 4, None]

        resistance = cohesion + np.maximum((rho_sat - (rho_s - rho_w)) * 9.81 * h0, 0.0) * np.tan(phi)
        initiation = (
            (rho_s - rho_w) * 9.81 * h0 * sin_t
            + rho_w * 9.81 * h0 * tan_t
            + rho_runoff * np.square(friction_velocity)
        )
        fs = np.clip(resistance / np.maximum(initiation, 1e-6), 0.0, 2.0)
        mean_fs[start:stop] = np.mean(fs, axis=1)
        unstable_fraction[start:stop] = np.mean(fs < 1.0, axis=1)
        for code, position in representative_positions.items():
            representative_fs[code][start:stop] = fs[:, position]
    return {"mean_fs": mean_fs, "unstable_fraction": unstable_fraction, **representative_fs}


def normalized_parameter_change_pct(
    key: str,
    value: float,
    baseline: np.ndarray,
    state: dict[str, np.ndarray | str],
) -> float:
    parameter_index = [item[0] for item in PARAMETERS].index(key)
    reference = float(baseline[parameter_index])
    return 100.0 * (value / reference - 1.0)


def run_oat(
    state: dict[str, np.ndarray | str],
    ranges: dict[str, tuple[float, float]],
    baseline: np.ndarray,
    levels: int,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    baseline_metrics = evaluate(baseline[None, :], state)
    rows: list[dict[str, float | str]] = []
    ranking: list[dict[str, float | str]] = []
    for param_index, (key, label, unit) in enumerate(PARAMETERS):
        lo, hi = ranges[key]
        values = np.linspace(lo, hi, levels)
        samples = np.repeat(baseline[None, :], levels, axis=0)
        samples[:, param_index] = values
        metrics = evaluate(samples, state)
        for level_index, value in enumerate(values):
            rows.append(
                {
                    "parameter": key,
                    "label": label,
                    "unit": unit.replace("$", ""),
                    "level": level_index + 1,
                    "value": float(value),
                    "input_change_pct": normalized_parameter_change_pct(key, float(value), baseline, state),
                    "mean_fs": float(metrics["mean_fs"][level_index]),
                    "mean_fs_change_pct": float(
                        100.0 * (metrics["mean_fs"][level_index] / baseline_metrics["mean_fs"][0] - 1.0)
                    ),
                    "unstable_fraction": float(metrics["unstable_fraction"][level_index]),
                    "unstable_fraction_change_percentage_points": float(
                        100.0 * (metrics["unstable_fraction"][level_index] - baseline_metrics["unstable_fraction"][0])
                    ),
                    **{
                        f"{code}_fs": float(metrics[code][level_index])
                        for code in ["R1", "R2", "N1"]
                        if code in metrics
                    },
                    **{
                        f"{code}_fs_change_pct": float(
                            100.0 * (metrics[code][level_index] / baseline_metrics[code][0] - 1.0)
                        )
                        for code in ["R1", "R2", "N1"]
                        if code in metrics and abs(float(baseline_metrics[code][0])) > 1e-12
                    },
                }
            )
        ranking.append(
            {
                "parameter": key,
                "label": label,
                "mean_fs_response_range": float(np.ptp(metrics["mean_fs"])),
                "unstable_fraction_response_range": float(np.ptp(metrics["unstable_fraction"])),
                "unstable_area_response_range_percentage_points": float(100.0 * np.ptp(metrics["unstable_fraction"])),
                "direction_with_increasing_parameter": (
                    "stabilizing" if metrics["mean_fs"][-1] > metrics["mean_fs"][0] else "destabilizing"
                ),
            }
        )
    ranking.sort(key=lambda row: float(row["unstable_fraction_response_range"]), reverse=True)
    return rows, ranking


def scale_unit_samples(unit_samples: np.ndarray, ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    scaled = np.empty_like(unit_samples, dtype="float64")
    for index, (key, _, _) in enumerate(PARAMETERS):
        lo, hi = ranges[key]
        scaled[:, index] = lo + unit_samples[:, index] * (hi - lo)
    return scaled


def index_estimates(y_a: np.ndarray, y_b: np.ndarray, y_ab: np.ndarray) -> tuple[float, float]:
    variance = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
    if variance <= 1e-15:
        return 0.0, 0.0
    first_order = 1.0 - float(np.mean((y_b - y_ab) ** 2)) / (2.0 * variance)
    total_order = float(np.mean((y_a - y_ab) ** 2)) / (2.0 * variance)
    return first_order, total_order


def bootstrap_indices(
    y_a: np.ndarray,
    y_b: np.ndarray,
    y_ab: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    n = y_a.size
    s1 = np.empty(bootstrap, dtype="float64")
    st = np.empty(bootstrap, dtype="float64")
    for i in range(bootstrap):
        idx = rng.integers(0, n, size=n)
        s1[i], st[i] = index_estimates(y_a[idx], y_b[idx], y_ab[idx])
    return tuple(float(v) for v in [*np.percentile(s1, [2.5, 97.5]), *np.percentile(st, [2.5, 97.5])])


def run_sobol(
    state: dict[str, np.ndarray | str],
    ranges: dict[str, tuple[float, float]],
    sobol_n: int,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    dimension = len(PARAMETERS)
    sampler = qmc.Sobol(d=dimension * 2, scramble=True, seed=seed)
    unit = sampler.random_base2(int(math.log2(sobol_n)))
    a = scale_unit_samples(unit[:, :dimension], ranges)
    b = scale_unit_samples(unit[:, dimension:], ranges)
    f_a = evaluate(a, state)
    f_b = evaluate(b, state)

    by_metric: dict[str, list[dict[str, float | str]]] = {"mean_fs": [], "unstable_fraction": []}
    hybrid_outputs: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(seed + 17)
    for param_index, (key, label, unit_label) in enumerate(PARAMETERS):
        ab = a.copy()
        ab[:, param_index] = b[:, param_index]
        f_ab = evaluate(ab, state)
        hybrid_outputs[key] = f_ab
        for metric in by_metric:
            s1, st = index_estimates(f_a[metric], f_b[metric], f_ab[metric])
            s1_lo, s1_hi, st_lo, st_hi = bootstrap_indices(
                f_a[metric], f_b[metric], f_ab[metric], bootstrap, rng
            )
            by_metric[metric].append(
                {
                    "metric": metric,
                    "parameter": key,
                    "label": label,
                    "unit": unit_label.replace("$", ""),
                    "first_order_index": s1,
                    "first_order_ci_low": s1_lo,
                    "first_order_ci_high": s1_hi,
                    "total_order_index": st,
                    "total_order_ci_low": st_lo,
                    "total_order_ci_high": st_hi,
                }
            )

    rows: list[dict[str, float | str]] = []
    for metric, metric_rows in by_metric.items():
        positive_total = sum(max(0.0, float(row["total_order_index"])) for row in metric_rows)
        for row in metric_rows:
            row["normalized_total_contribution_pct"] = (
                100.0 * max(0.0, float(row["total_order_index"])) / positive_total if positive_total > 0 else 0.0
            )
            rows.append(row)
    convergence_rows: list[dict[str, float | str]] = []
    sample_sizes = [2**power for power in range(6, int(math.log2(sobol_n)) + 1)]
    for sample_size in sample_sizes:
        for param_index, (key, label, _) in enumerate(PARAMETERS):
            f_ab = hybrid_outputs[key]
            for metric in ["mean_fs", "unstable_fraction"]:
                s1, st = index_estimates(
                    f_a[metric][:sample_size], f_b[metric][:sample_size], f_ab[metric][:sample_size]
                )
                convergence_rows.append(
                    {
                        "sample_size": sample_size,
                        "model_evaluations": sample_size * (2 + dimension),
                        "metric": metric,
                        "parameter": key,
                        "label": label,
                        "first_order_index": s1,
                        "total_order_index": st,
                    }
                )
    return rows, convergence_rows


def write_dict_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows were generated for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_oat(rows: list[dict[str, float | str]], baseline: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    for panel_index, (axis, (key, label, unit)) in enumerate(zip(axes.ravel(), PARAMETERS)):
        selected = [row for row in rows if row["parameter"] == key]
        x = np.asarray([float(row["value"]) for row in selected])
        mean_fs = np.asarray([float(row["mean_fs"]) for row in selected])
        unstable = 100.0 * np.asarray([float(row["unstable_fraction"]) for row in selected])
        second = axis.twinx()
        axis.plot(x, mean_fs, color=COLORS["blue"], marker="o", ms=3.2, lw=1.4)
        second.plot(x, unstable, color=COLORS["red"], marker="s", ms=3.0, lw=1.3)
        axis.axvline(baseline[panel_index], color=COLORS["grey"], lw=0.8, ls="--")
        axis.set_title(f"{chr(97 + panel_index)}  {label}", loc="left", fontweight="bold")
        axis.set_xlabel(f"Value ({unit})")
        axis.set_ylabel("Mean Fs", color=COLORS["blue"])
        second.set_ylabel("Unstable area (%)", color=COLORS["red"])
        axis.tick_params(axis="y", colors=COLORS["blue"])
        second.tick_params(axis="y", colors=COLORS["red"])
        axis.spines["top"].set_visible(False)
        second.spines["top"].set_visible(False)
        axis.grid(axis="y", color="#E6E9EC", lw=0.6)
    for axis in axes.ravel()[len(PARAMETERS):]:
        axis.set_visible(False)
    save_figure(fig, "parameter_sensitivity_oat")
    plt.close(fig)


def representative_oat_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    for row in rows:
        for code in ["R1", "R2", "N1"]:
            if f"{code}_fs" not in row:
                continue
            output.append(
                {
                    "parameter": row["parameter"],
                    "label": row["label"],
                    "point": code,
                    "input_change_pct": row["input_change_pct"],
                    "fs": row[f"{code}_fs"],
                    "fs_change_pct": row[f"{code}_fs_change_pct"],
                }
            )
    return output


def plot_representative_oat(rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=False, constrained_layout=True)
    point_colors = {"R1": COLORS["red"], "R2": COLORS["gold"], "N1": COLORS["blue"]}
    for panel_index, (axis, (key, label, _)) in enumerate(zip(axes.ravel(), PARAMETERS)):
        for code in ["R1", "R2", "N1"]:
            selected = [row for row in rows if row["parameter"] == key and row["point"] == code]
            axis.plot(
                [float(row["input_change_pct"]) for row in selected],
                [float(row["fs_change_pct"]) for row in selected],
                color=point_colors[code],
                marker="o",
                ms=3.0,
                lw=1.3,
                label=code,
            )
        axis.axhline(0.0, color="#333333", lw=0.7, ls="--")
        axis.axvline(0.0, color=COLORS["grey"], lw=0.7, ls=":")
        axis.set_title(f"{chr(97 + panel_index)}  {label}", loc="left", fontweight="bold")
        axis.set_xlabel("Input change (%)")
        axis.set_ylabel("Fs change (%)")
        axis.grid(color="#E6E9EC", lw=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes.ravel()[len(PARAMETERS):]:
        axis.set_visible(False)
    axes[0, 0].legend(ncol=3, loc="best")
    save_figure(fig, "representative_point_sensitivity_oat")
    plt.close(fig)


def plot_contribution(rows: list[dict[str, float | str]]) -> None:
    unstable_rows = [row for row in rows if row["metric"] == "unstable_fraction"]
    order = [
        row["parameter"]
        for row in sorted(unstable_rows, key=lambda row: float(row["normalized_total_contribution_pct"]))
    ]
    labels = {key: label for key, label, _ in PARAMETERS}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, constrained_layout=True)
    for panel_index, (axis, metric, title) in enumerate(
        zip(axes, ["mean_fs", "unstable_fraction"], ["Mean Fs", "Unstable-area fraction"])
    ):
        lookup = {row["parameter"]: row for row in rows if row["metric"] == metric}
        values = np.asarray([float(lookup[key]["normalized_total_contribution_pct"]) for key in order])
        first = np.asarray([max(0.0, float(lookup[key]["first_order_index"])) for key in order])
        total = np.asarray([max(0.0, float(lookup[key]["total_order_index"])) for key in order])
        total_sum = float(np.sum(total))
        first_pct = 100.0 * first / total_sum if total_sum > 0 else np.zeros_like(first)
        y = np.arange(len(order))
        axis.barh(y, values, color=COLORS["blue"] if panel_index == 0 else COLORS["red"], alpha=0.82)
        axis.scatter(first_pct, y, s=22, facecolor="white", edgecolor="#222222", linewidth=0.8, zorder=3, label="First-order component")
        axis.set_yticks(y, [labels[key] for key in order])
        axis.set_xlabel("Normalized contribution (%)")
        axis.set_title(f"{chr(97 + panel_index)}  {title}", loc="left", fontweight="bold")
        axis.grid(axis="x", color="#E6E9EC", lw=0.6)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_figure(fig, "parameter_contribution_sobol")
    plt.close(fig)


def plot_sobol_convergence(rows: list[dict[str, float | str]]) -> None:
    selected = [row for row in rows if row["metric"] == "unstable_fraction"]
    fig, axis = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    colors = [COLORS["red"], COLORS["gold"], COLORS["blue"], COLORS["green"], COLORS["grey"], "#765A9C"]
    for (key, label, _), color in zip(PARAMETERS, colors):
        values = [row for row in selected if row["parameter"] == key]
        axis.plot(
            [int(row["sample_size"]) for row in values],
            [float(row["total_order_index"]) for row in values],
            marker="o",
            ms=3.2,
            lw=1.3,
            color=color,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Base sample size")
    axis.set_ylabel("Total-order index")
    axis.set_title("Convergence of global parameter sensitivity", loc="left", fontweight="bold")
    axis.grid(color="#E6E9EC", lw=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, "parameter_contribution_sobol_convergence")
    plt.close(fig)


def polynomial_design(x: np.ndarray, degree: int) -> np.ndarray:
    columns = [np.ones(x.shape[0], dtype="float64")]
    for current_degree in range(1, degree + 1):
        for indices in combinations_with_replacement(range(x.shape[1]), current_degree):
            columns.append(np.prod(x[:, indices], axis=1))
    return np.column_stack(columns)


def fit_polynomial_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    ridge: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    split = max(1, int(0.8 * x.shape[0]))
    train_idx, test_idx = order[:split], order[split:]
    design_train = polynomial_design(x[train_idx], degree)
    penalty = np.eye(design_train.shape[1], dtype="float64") * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design_train.T @ design_train + penalty,
        design_train.T @ y[train_idx],
    )
    prediction = polynomial_design(x[test_idx], degree) @ coefficients
    residual = float(np.sum((y[test_idx] - prediction) ** 2))
    total = float(np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2))
    r2 = 1.0 - residual / total if total > 1e-15 else 1.0
    return coefficients, train_idx, test_idx, r2


def run_shap_surrogate(
    state: dict[str, np.ndarray | str],
    ranges: dict[str, tuple[float, float]],
    seed: int,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], dict[str, float | int | str]]:
    cfg = CFG.get("analysis", {})
    requested = max(256, int(cfg.get("shap_surrogate_samples", 2048)))
    sample_count = 2 ** int(math.ceil(math.log2(requested)))
    display_count = max(100, int(cfg.get("shap_display_samples", 600)))
    permutations = max(16, int(cfg.get("shap_permutations", 64)))
    degree = max(1, min(4, int(cfg.get("shap_polynomial_degree", 3))))
    ridge = max(0.0, float(cfg.get("shap_ridge", 1e-8)))

    sampler = qmc.Sobol(d=len(PARAMETERS), scramble=True, seed=seed + 101)
    unit_samples = sampler.random_base2(int(math.log2(sample_count)))
    physical_samples = scale_unit_samples(unit_samples, ranges)
    response = evaluate(physical_samples, state)["mean_fs"]
    standardized = 2.0 * unit_samples - 1.0
    coefficients, train_idx, test_idx, r2 = fit_polynomial_surrogate(
        standardized, response, degree, ridge, seed + 102
    )
    if r2 < 0.95:
        raise RuntimeError(
            f"SHAP surrogate accuracy is insufficient (test R2={r2:.4f}). "
            "Increase polynomial degree/samples before interpreting SHAP values."
        )

    rng = np.random.default_rng(seed + 103)
    if test_idx.size >= display_count:
        display_idx = rng.choice(test_idx, size=display_count, replace=False)
    else:
        display_idx = rng.choice(np.arange(sample_count), size=min(display_count, sample_count), replace=False)
    x_display = standardized[display_idx]
    physical_display = physical_samples[display_idx]
    shap_values = np.zeros_like(x_display)
    base_prediction = np.zeros(x_display.shape[0], dtype="float64")

    def predict(values: np.ndarray) -> np.ndarray:
        return polynomial_design(values, degree) @ coefficients

    for _ in range(permutations):
        background = standardized[rng.choice(train_idx, size=x_display.shape[0], replace=True)]
        state_values = background.copy()
        previous = predict(state_values)
        base_prediction += previous
        for feature_index in rng.permutation(len(PARAMETERS)):
            state_values[:, feature_index] = x_display[:, feature_index]
            current = predict(state_values)
            shap_values[:, feature_index] += current - previous
            previous = current
    shap_values /= permutations
    base_prediction /= permutations
    prediction = predict(x_display)
    local_residual = prediction - (base_prediction + np.sum(shap_values, axis=1))

    value_rows: list[dict[str, float | str]] = []
    importance_rows: list[dict[str, float | str]] = []
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    for feature_index, (key, label, unit) in enumerate(PARAMETERS):
        importance_rows.append(
            {
                "parameter": key,
                "label": label,
                "unit": unit.replace("$", ""),
                "mean_absolute_shap": float(mean_abs[feature_index]),
                "normalized_contribution_pct": float(100.0 * mean_abs[feature_index] / np.sum(mean_abs)),
            }
        )
        for sample_index in range(x_display.shape[0]):
            value_rows.append(
                {
                    "sample": sample_index + 1,
                    "parameter": key,
                    "label": label,
                    "feature_value": float(physical_display[sample_index, feature_index]),
                    "normalized_feature_value": float(unit_samples[display_idx[sample_index], feature_index]),
                    "shap_value": float(shap_values[sample_index, feature_index]),
                }
            )
    importance_rows.sort(key=lambda row: float(row["mean_absolute_shap"]), reverse=True)
    diagnostics: dict[str, float | int | str] = {
        "target": "critical-time spatial mean Fs",
        "surrogate": f"degree-{degree} polynomial response surface",
        "surrogate_samples": sample_count,
        "display_samples": int(x_display.shape[0]),
        "shap_permutations": permutations,
        "test_r2": r2,
        "maximum_local_additivity_residual": float(np.max(np.abs(local_residual))),
    }
    return value_rows, importance_rows, diagnostics


def plot_shap_summary(
    value_rows: list[dict[str, float | str]],
    importance_rows: list[dict[str, float | str]],
    diagnostics: dict[str, float | int | str],
) -> None:
    order = [str(row["parameter"]) for row in importance_rows]
    labels = {key: label for key, label, _ in PARAMETERS}
    rng = np.random.default_rng(20260909)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.45, 1.0]}, constrained_layout=True)
    scatter = None
    for y_index, key in enumerate(reversed(order)):
        selected = [row for row in value_rows if row["parameter"] == key]
        shap_values = np.asarray([float(row["shap_value"]) for row in selected])
        feature_values = np.asarray([float(row["normalized_feature_value"]) for row in selected])
        jitter = rng.normal(0.0, 0.085, size=shap_values.size)
        scatter = axes[0].scatter(
            shap_values,
            y_index + jitter,
            c=feature_values,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=8,
            alpha=0.72,
            linewidths=0,
        )
    axes[0].axvline(0.0, color="#777777", lw=0.8)
    axes[0].set_yticks(np.arange(len(order)), [labels[key] for key in reversed(order)])
    axes[0].set_xlabel("SHAP value (impact on mean Fs)")
    axes[0].set_title("a  SHAP summary", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#E6E9EC", lw=0.6)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02, fraction=0.05)
        colorbar.set_label("Feature value")
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"])

    x = np.arange(len(importance_rows))
    importance = np.asarray([float(row["mean_absolute_shap"]) for row in importance_rows])
    bars = axes[1].bar(x, importance, color=COLORS["red"], alpha=0.86, width=0.72)
    axes[1].set_xticks(x, [labels[str(row["parameter"])] for row in importance_rows], rotation=35, ha="right")
    axes[1].set_ylabel("Mean |SHAP value|")
    axes[1].set_title("b  Mean parameter contribution", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#E6E9EC", lw=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for bar, value in zip(bars, importance):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
    axes[1].text(
        0.98,
        0.96,
        f"Surrogate test $R^2$ = {float(diagnostics['test_r2']):.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    save_figure(fig, "parameter_contribution_shap")
    plt.close(fig)


def plot_representative_fs_rainfall(rows: list[dict[str, str]]) -> None:
    rainfall = np.asarray([float(row["rain_mm"]) for row in rows])
    hours = np.arange(len(rows), dtype="float64")
    peak_index = int(np.argmax(rainfall))
    r1_fs = np.asarray([float(row["R1_fs"]) for row in rows])
    r1_min_index = int(np.argmin(r1_fs))
    r1_lag = r1_min_index - peak_index
    fig, axis = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    rain_axis = axis.twinx()
    rain_axis.bar(hours, rainfall, width=0.62, color="#69BCE6", edgecolor="#2579A4", linewidth=0.5, alpha=0.55, label="Hourly rainfall")
    rain_axis.set_ylim(max(float(np.max(rainfall)) * 1.28, 1.0), 0.0)
    rain_axis.set_ylabel("Hourly rainfall (mm)")
    rain_axis.spines["top"].set_visible(False)

    point_colors = {"R1": COLORS["red"], "R2": COLORS["gold"], "N1": COLORS["green"]}
    for code in ["R1", "R2", "N1"]:
        axis.plot(
            hours,
            [float(row[f"{code}_fs"]) for row in rows],
            color=point_colors[code],
            lw=1.7,
            marker="o",
            ms=3.2,
            label=code,
            zorder=4,
        )
    axis.axhline(1.0, color="#222222", lw=0.9, ls="--", label="Fs = 1")
    axis.axvline(float(peak_index), color=COLORS["grey"], lw=0.9, ls=":")
    if r1_min_index != peak_index:
        axis.axvline(float(r1_min_index), color=COLORS["red"], lw=0.8, ls=":")
    axis.set_xlabel("Time from simulation start (h)")
    axis.set_ylabel("Safety factor")
    axis.set_xlim(-0.4, len(rows) - 0.6)
    fs_values = [float(row[f"{code}_fs"]) for row in rows for code in ["R1", "R2", "N1"]]
    axis.set_ylim(max(0.0, min(fs_values) - 0.12), max(fs_values) + 0.18)
    axis.set_title("Safety-factor response to hourly rainfall", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#E6E9EC", lw=0.6)
    axis.spines["top"].set_visible(False)
    line_handles, line_labels = axis.get_legend_handles_labels()
    rain_handles, rain_labels = rain_axis.get_legend_handles_labels()
    axis.legend(
        rain_handles + line_handles,
        rain_labels + line_labels,
        ncol=5,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        handlelength=1.6,
        columnspacing=1.0,
    )
    axis.text(
        peak_index + 0.12,
        1.04,
        f"Rainfall peak to R1 minimum\nlag = {r1_lag:+d} h",
        fontsize=7,
        ha="left",
        va="bottom",
    )
    save_figure(fig, "representative_point_fs_rainfall")
    plt.close(fig)


def plot_time_series(
    rows: list[dict[str, str]], representative_rows: list[dict[str, str]]
) -> dict[str, float | str]:
    times = [datetime.fromisoformat(row["datetime"]) for row in rows]
    rainfall = np.asarray([float(row["rain_mm"]) for row in rows])
    mean_fs = np.asarray([float(row["mean_fs"]) for row in rows])
    p05 = np.asarray([float(row["sample_p05_fs"]) for row in rows])
    p95 = np.asarray([float(row["sample_p95_fs"]) for row in rows])
    unstable = 100.0 * np.asarray([float(row["unstable_fraction"]) for row in rows])
    area = np.asarray([float(row["unstable_area_km2"]) for row in rows])

    peak_rain_index = int(np.argmax(rainfall))
    min_fs_index = int(np.argmin(mean_fs))
    max_unstable_index = int(np.argmax(unstable))
    lag_h = (times[min_fs_index] - times[peak_rain_index]).total_seconds() / 3600.0
    positive_rain = np.flatnonzero(rainfall > 0)
    pre_event_index = max(0, int(positive_rain[0]) - 1) if positive_rain.size else 0
    baseline_unstable = float(unstable[pre_event_index])
    peak_excess_unstable = float(unstable[max_unstable_index] - baseline_unstable)
    baseline_area = float(area[pre_event_index])
    peak_excess_area = float(area[max_unstable_index] - baseline_area)

    representative_fs = {
        code: np.asarray([float(row[f"{code}_fs"]) for row in representative_rows])
        for code in ["R1", "R2", "N1"]
    }
    point_lags: dict[str, float] = {}
    for code, values in representative_fs.items():
        point_lags[code] = (times[int(np.argmin(values))] - times[peak_rain_index]).total_seconds() / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.8), sharex=True, constrained_layout=True)
    axes[0].bar(times, rainfall, width=0.032, color="#8AAFC7", edgecolor="#5A839D", linewidth=0.5)
    axes[0].set_ylabel("Rainfall\n(mm h$^{-1}$)")
    axes[0].set_title("a  Rainfall forcing", loc="left", fontweight="bold")

    point_colors = {"R1": COLORS["red"], "R2": COLORS["gold"], "N1": COLORS["green"]}
    for code in ["R1", "R2", "N1"]:
        axes[1].plot(times, representative_fs[code], color=point_colors[code], lw=1.5, marker="o", ms=2.8, label=code)
    axes[1].axhline(1.0, color="#222222", lw=0.9, ls="--", label="Fs = 1")
    axes[1].set_ylabel("Safety factor")
    axes[1].set_title("b  Representative-point stability", loc="left", fontweight="bold")
    axes[1].legend(ncol=4, loc="best")

    axes[2].fill_between(times, p05, p95, color=COLORS["blue"], alpha=0.16, label="Sampled 5th-95th percentile")
    axes[2].plot(times, mean_fs, color=COLORS["blue"], lw=1.6, marker="o", ms=3.0, label="Domain mean Fs")
    axes[2].axhline(1.0, color="#222222", lw=0.9, ls="--", label="Fs = 1")
    axes[2].set_ylabel("Safety factor")
    axes[2].set_title("c  Regional stability response", loc="left", fontweight="bold")
    axes[2].legend(ncol=3, loc="lower left")

    axes[3].plot(times, unstable, color=COLORS["red"], lw=1.6, marker="s", ms=3.0)
    axes[3].fill_between(times, 0.0, unstable, color=COLORS["red"], alpha=0.14)
    axes[3].set_ylabel("Unstable area (%)")
    axes[3].set_title("d  Spatial extent of instability", loc="left", fontweight="bold")
    axes[3].set_xlabel("Date and time (Beijing time)")
    for axis in axes:
        axis.axvline(times[peak_rain_index], color=COLORS["gold"], lw=0.9, ls=":")
        axis.grid(axis="y", color="#E6E9EC", lw=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.autofmt_xdate(rotation=0)
    save_figure(fig, "safety_factor_time_series")
    plt.close(fig)
    return {
        "peak_rain_time": times[peak_rain_index].isoformat(sep=" "),
        "minimum_mean_fs_time": times[min_fs_index].isoformat(sep=" "),
        "maximum_unstable_area_time": times[max_unstable_index].isoformat(sep=" "),
        "lag_peak_rain_to_minimum_mean_fs_h": lag_h,
        "minimum_mean_fs": float(mean_fs[min_fs_index]),
        "maximum_unstable_fraction": float(unstable[max_unstable_index] / 100.0),
        "maximum_unstable_area_km2": float(area[max_unstable_index]),
        "initial_unstable_fraction": baseline_unstable / 100.0,
        "initial_unstable_area_km2": baseline_area,
        "peak_excess_unstable_percentage_points": peak_excess_unstable,
        "peak_excess_unstable_area_km2": peak_excess_area,
        "pre_event_time": times[pre_event_index].isoformat(sep=" "),
        **{f"lag_peak_rain_to_minimum_{code}_fs_h": lag for code, lag in point_lags.items()},
    }


def write_and_plot_stability_stages(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    times = [datetime.fromisoformat(row["datetime"]) for row in rows]
    rainfall = np.asarray([float(row["rain_mm"]) for row in rows])
    unstable = np.asarray([float(row["unstable_fraction"]) for row in rows])
    positive_rain = np.flatnonzero(rainfall > 0)
    pre_event = max(0, int(positive_rain[0]) - 1) if positive_rain.size else 0
    candidates = [
        ("Pre-event", pre_event),
        ("Rainfall peak", int(np.argmax(rainfall))),
        ("Maximum unstable extent", int(np.argmax(unstable))),
        ("Simulation end", len(rows) - 1),
    ]
    stages: list[tuple[str, int]] = []
    for label, index in candidates:
        existing = next((i for i, (_, idx) in enumerate(stages) if idx == index), None)
        if existing is None:
            stages.append((label, index))
        else:
            old_label, _ = stages[existing]
            stages[existing] = (f"{old_label} / {label.lower()}", index)

    output: list[dict[str, float | str]] = []
    fields = [
        ("Fs < 1.0", "fraction_fs_lt_1"),
        ("1.0 <= Fs < 1.25", "fraction_fs_1_to_1p25"),
        ("1.25 <= Fs < 1.5", "fraction_fs_1p25_to_1p5"),
        ("Fs >= 1.5", "fraction_fs_ge_1p5"),
    ]
    for label, index in stages:
        row: dict[str, float | str] = {"stage": label, "datetime": times[index].isoformat(sep=" ")}
        for display, field in fields:
            row[display] = 100.0 * float(rows[index][field])
        output.append(row)

    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    bottom = np.zeros(len(output))
    colors = [COLORS["red"], COLORS["gold"], "#8AAFC7", COLORS["green"]]
    for (display, _), color in zip(fields, colors):
        values = np.asarray([float(row[display]) for row in output])
        axis.bar(np.arange(len(output)), values, bottom=bottom, color=color, width=0.68, label=display)
        bottom += values
    axis.set_xticks(np.arange(len(output)), [str(row["stage"]) for row in output])
    axis.set_ylabel("Area proportion (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Staged evolution of regional slope stability", loc="left", fontweight="bold")
    axis.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E6E9EC", lw=0.6)
    axis.set_axisbelow(True)
    save_figure(fig, "staged_stability_comparison")
    plt.close(fig)
    return output


def main() -> None:
    out_dir().mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    state = load_state()
    ranges, baseline, levels, sobol_n, bootstrap = analysis_settings(state)
    seed = int(CFG.get("analysis", {}).get("random_seed", 20260908))

    oat_rows, ranking_rows = run_oat(state, ranges, baseline, levels)
    contribution_rows, convergence_rows = run_sobol(state, ranges, sobol_n, bootstrap, seed)
    time_rows = read_csv_rows(source_dir() / f"{slug()}_safety_factor_time_series.csv")
    representative_time_rows = read_csv_rows(source_dir() / f"{slug()}_representative_point_time_series.csv")
    temporal_summary = plot_time_series(time_rows, representative_time_rows)
    plot_representative_fs_rainfall(representative_time_rows)
    stage_rows = write_and_plot_stability_stages(time_rows)
    representative_sensitivity_rows = representative_oat_rows(oat_rows)
    shap_rows, shap_importance_rows, shap_diagnostics = run_shap_surrogate(state, ranges, seed)

    write_dict_rows(out_dir() / "parameter_sensitivity_oat.csv", oat_rows)
    write_dict_rows(out_dir() / "parameter_sensitivity_ranking.csv", ranking_rows)
    write_dict_rows(out_dir() / "parameter_contribution_sobol.csv", contribution_rows)
    write_dict_rows(out_dir() / "parameter_contribution_sobol_convergence.csv", convergence_rows)
    write_dict_rows(out_dir() / "representative_point_sensitivity_oat.csv", representative_sensitivity_rows)
    write_dict_rows(out_dir() / "parameter_contribution_shap_values.csv", shap_rows)
    write_dict_rows(out_dir() / "parameter_contribution_shap_importance.csv", shap_importance_rows)
    write_dict_rows(out_dir() / "staged_stability_comparison.csv", stage_rows)
    plot_oat(oat_rows, baseline)
    plot_representative_oat(representative_sensitivity_rows)
    plot_contribution(contribution_rows)
    plot_sobol_convergence(convergence_rows)
    plot_shap_summary(shap_rows, shap_importance_rows, shap_diagnostics)

    metadata = {
        "project": slug(),
        "critical_time": state["critical_time"],
        "sample_cells": int(len(np.asarray(state["base_fs"]))),
        "oat_levels": levels,
        "sobol_base_samples": sobol_n,
        "sobol_model_evaluations": sobol_n * (2 + len(PARAMETERS)),
        "sobol_bootstrap_replicates": bootstrap,
        "mechanics_model": CFG.get("stability", {}).get("mechanics_model", "reference_equation_9"),
        "parameter_ranges": ranges,
        "shap_surrogate": shap_diagnostics,
        "temporal_summary": temporal_summary,
    }
    (out_dir() / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    most_sensitive = ranking_rows[0]
    unstable_contrib = [row for row in contribution_rows if row["metric"] == "unstable_fraction"]
    dominant = max(unstable_contrib, key=lambda row: float(row["normalized_total_contribution_pct"]))
    initial_unstable_pct = 100.0 * float(temporal_summary["initial_unstable_fraction"])
    peak_unstable_pct = 100.0 * float(temporal_summary["maximum_unstable_fraction"])
    baseline_note = (
        f"降雨前已有 {initial_unstable_pct:.2f}% 的有效区处于 Fs < 1，峰时为 {peak_unstable_pct:.2f}%，"
        f"相对降雨前状态增加 {float(temporal_summary['peak_excess_unstable_percentage_points']):.2f} 个百分点"
        f"（约 {float(temporal_summary['peak_excess_unstable_area_km2']):.2f} km2）。"
    )
    readme = out_dir() / "README_参数与时序分析说明.md"
    readme.write_text(
        "\n".join(
            [
                "# 参数敏感性、参数贡献与安全系数时序分析",
                "",
                f"临界分析时刻：{state['critical_time']}。该时刻对应全域不稳定面积比例最大值。",
                f"空间样本数：{metadata['sample_cells']}；Sobol 基础样本数：{sobol_n}；模型评价次数：{metadata['sobol_model_evaluations']}。",
                "",
                "## 1 参数敏感性分析",
                "",
                "采用单因素扰动法，在保持临界时刻流速、DEM 坡度和土地利用空间分区不变的情况下，依次改变天然密度、饱和密度、饱和内摩擦角、饱和黏聚力和冲刷径流密度。五个不确定参数及其范围与参考文献保持一致。",
                "同时按输入参数相对变化率与 R1、R2、N1 的 Fs 相对变化率绘图，以便比较不同响应位置对同一参数扰动的差异。坡度由输入 DEM 确定，不作为待率定不确定参数进行人为扰动。",
                f"按不稳定面积响应范围排序，当前最敏感参数为：{most_sensitive['label']}。",
                "",
                "## 2 参数贡献分析",
                "",
                "采用扰动后的 Sobol 准蒙特卡洛样本和 Jansen 估计量计算一阶与总效应指数。归一化总效应贡献用于图示，各参数原始指数及 95% 自助法区间保存在 CSV 中。总效应包含参数自身及其与其他参数的交互作用。",
                "同步输出不同基础样本量下的总效应指数，用于判断 Sobol 排序是否趋于稳定。这里的“贡献”是给定参数范围内对模型输出方差的贡献，不是观测意义上的因果贡献。",
                f"另以 {shap_diagnostics['surrogate_samples']} 组力学模型样本拟合三阶响应面，并采用 Monte Carlo Shapley 置换近似解释临界时刻平均 Fs。代理模型测试集 R2={float(shap_diagnostics['test_r2']):.4f}；SHAP 图解释的是代理模型对原力学模型的近似，不替代 Sobol 全局敏感性。",
                f"对不稳定面积比例贡献最大的参数为：{dominant['label']}，归一化总效应贡献为 {float(dominant['normalized_total_contribution_pct']):.2f}%。",
                "",
                "## 3 安全系数时序变化",
                "",
                "时序结果同时展示逐时降雨、R1/R2/N1 代表点 Fs、全域平均 Fs、均匀空间样本的 5%-95% 分位区间及不稳定面积比例。虚线 Fs=1 为启动判别参考值。另按降雨前、降雨峰值/最大不稳定范围和计算结束时刻统计四级稳定性面积组成。",
                "同时输出单坐标框论文图，将倒置小时降雨柱与 R1/R2/N1 的 Fs 曲线叠加，便于直接识别启动阈值、峰值响应和时间差。",
                f"降雨峰值至全域平均 Fs 最低值的时间差为 {float(temporal_summary['lag_peak_rain_to_minimum_mean_fs_h']):.2f} h。",
                f"比较基准采用降雨开始前最后一个时刻（{temporal_summary['pre_event_time']}），而不是计算序列首时刻。",
                baseline_note,
                "因此，解释降雨效应时应同时报告绝对不稳定比例和相对初始状态的增量；在完成初始稳定性校准前，不应把峰时全部 Fs < 1 区域都归因于本次降雨。",
                "",
                "## 解释边界",
                "",
                "敏感性与 Sobol 指数是以临界时刻水动力状态为条件的启动稳定性参数分析，不代表降雨来源或完整水动力参数的贡献。当前贡献比例用于比较本次设定参数范围内的相对重要性，不能脱离参数范围解释为普适常数。",
                "安全系数采用参考文献式（3）至式（9）：坡度控制浮重与渗流力，平均流速经 a=0.6 转为摩阻流速并进入冲刷项，土地利用分区控制 H、c' 和 phi'。因此时序突变由流速过程驱动；该模型不是 TRIGRS 入渗解析解，不应把时间差解释为地下孔压传播滞后。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] Parameter and temporal analyses written: {out_dir()}")
    print(f"[INFO] Critical time={state['critical_time']}; sample cells={metadata['sample_cells']}")
    print(f"[INFO] OAT most sensitive={most_sensitive['label']}")
    print(
        f"[INFO] Dominant Sobol contribution for unstable area={dominant['label']} "
        f"({float(dominant['normalized_total_contribution_pct']):.2f}%)"
    )
    print(
        f"[INFO] Peak-rain to minimum-mean-Fs lag="
        f"{float(temporal_summary['lag_peak_rain_to_minimum_mean_fs_h']):.2f} h"
    )


if __name__ == "__main__":
    main()
