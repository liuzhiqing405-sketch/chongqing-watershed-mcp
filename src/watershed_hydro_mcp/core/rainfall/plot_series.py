from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..common import event_slug, event_times
from ..rainfall.station_rainfall import read_station_observations


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT

CFG = LazyConfig()


def read_domain_series(path: Path) -> tuple[list[datetime], np.ndarray]:
    times = []
    values = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            times.append(datetime.fromisoformat(row["datetime_beijing"]))
            values.append(float(row["rain_mm"]))
    if not times:
        raise RuntimeError(f"No readable rainfall was found in {path}")
    return times, np.asarray(values, dtype="float64")


def main() -> None:
    slug = event_slug(CFG)
    rain_cfg = CFG.get("rainfall", {})
    observations_path = Path(rain_cfg.get("station_observations_csv") or "")
    domain_path = Path(rain_cfg.get("csv") or "")
    if not observations_path.is_file():
        raise RuntimeError(f"Station observation table is missing: {observations_path}")
    if not domain_path.is_file():
        raise RuntimeError(f"Area-weighted rainfall series is missing: {domain_path}")

    observations = read_station_observations(observations_path)
    domain_times, domain_rainfall = read_domain_series(domain_path)
    start, event_end, simulation_end = event_times(CFG)
    figure_dir = ROOT / "inputs" / "rainfall" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(observations.keys)))
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), dpi=300, gridspec_kw={"height_ratios": [2.1, 1.0]}, constrained_layout=True)

    axis = axes[0]
    axis.axvspan(start, event_end, color="#E6B84A", alpha=0.12, linewidth=0, label="Main event window")
    for index, station_key in enumerate(observations.keys):
        axis.plot(
            observations.datetimes,
            observations.rainfall_mm[index],
            color=colors[index],
            lw=1.0,
            alpha=0.72,
            marker="o",
            ms=2.2,
            label=station_key,
        )
    axis.plot(domain_times, domain_rainfall, color="#111111", lw=2.6, marker="o", ms=3.2, label="IDW area mean", zorder=5)
    axis.set_xlim(start, simulation_end)
    axis.set_ylabel("Hourly rainfall (mm h$^{-1}$)")
    axis.set_title("a  Multi-station rainfall and IDW area mean", loc="left", fontweight="bold")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.grid(axis="y", color="#DDE2E6", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=4, loc="upper left", frameon=False)

    totals = np.sum(observations.rainfall_mm, axis=1)
    labels = observations.keys + ["IDW mean"]
    total_values = np.r_[totals, np.sum(domain_rainfall)]
    bar_colors = list(colors) + ["#111111"]
    bars = axes[1].bar(np.arange(len(labels)), total_values, color=bar_colors, width=0.72)
    axes[1].set_xticks(np.arange(len(labels)), labels)
    axes[1].set_ylabel("Event rainfall (mm)")
    axes[1].set_title("b  Event totals", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#DDE2E6", linewidth=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for bar, value in zip(bars, total_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    out_png = figure_dir / f"{slug}_rainfall_time_series.png"
    out_pdf = figure_dir / f"{slug}_rainfall_time_series.pdf"
    out_csv = figure_dir / f"{slug}_rainfall_time_series_stats.csv"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["series", "station_id", "station_name", "event_total_mm", "peak_mm_h", "peak_time"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, station_key in enumerate(observations.keys):
            peak_index = int(np.argmax(observations.rainfall_mm[index]))
            writer.writerow(
                {
                    "series": station_key,
                    "station_id": observations.station_ids[index],
                    "station_name": observations.names[index],
                    "event_total_mm": f"{totals[index]:.4f}",
                    "peak_mm_h": f"{observations.rainfall_mm[index, peak_index]:.4f}",
                    "peak_time": observations.datetimes[peak_index].strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        peak_index = int(np.argmax(domain_rainfall))
        writer.writerow(
            {
                "series": "IDW area mean",
                "station_id": "",
                "station_name": "",
                "event_total_mm": f"{np.sum(domain_rainfall):.4f}",
                "peak_mm_h": f"{domain_rainfall[peak_index]:.4f}",
                "peak_time": domain_times[peak_index].strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    print(f"[OK] Multi-station rainfall figure: {out_png}")
    print(f"[OK] Multi-station rainfall statistics: {out_csv}")
    print(
        f"[INFO] Stations={len(observations.keys)}; IDW area-mean total={np.sum(domain_rainfall):.3f} mm; "
        f"peak={np.max(domain_rainfall):.3f} mm/h"
    )


if __name__ == "__main__":
    main()
