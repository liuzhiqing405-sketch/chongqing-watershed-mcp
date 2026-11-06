from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class SwmmRunoffSeries:
    subcatchment_names: list[str]
    datetimes: list[datetime]
    runoff_cms: np.ndarray
    report_step_seconds: int


FLOW_TO_CMS = {
    0: 0.028316846592,       # cfs
    1: 0.0000630901964,     # US gpm
    2: 0.0438126365741,     # million US gallons/day
    3: 1.0,                 # cms
    4: 0.001,               # L/s
    5: 0.0115740740741,     # million L/day
}


def _read_i32(stream) -> int:
    raw = stream.read(4)
    if len(raw) != 4:
        raise RuntimeError("Unexpected end of SWMM output file.")
    return struct.unpack("<i", raw)[0]


def _read_names(stream, offset: int, count: int) -> list[str]:
    stream.seek(offset)
    names = []
    for _ in range(count):
        length = _read_i32(stream)
        if length < 0 or length > 4096:
            raise RuntimeError(f"Invalid SWMM object-name length: {length}")
        names.append(stream.read(length).decode("utf-8", errors="replace"))
    return names


def read_subcatchment_runoff(path: str | Path) -> SwmmRunoffSeries:
    """Read subcatchment runoff rates from a SWMM 5 binary output file."""
    output_path = Path(path)
    with output_path.open("rb") as stream:
        stream.seek(-24, 2)
        id_pos, obj_prop_pos, results_pos, periods, error_code, ending_magic = struct.unpack(
            "<6i", stream.read(24)
        )
        stream.seek(0)
        starting_magic, version, flow_units, n_subcatch, n_nodes, n_links, n_pollutants = struct.unpack(
            "<7i", stream.read(28)
        )
        if starting_magic != ending_magic:
            raise RuntimeError("Invalid SWMM output file: magic numbers do not match.")
        if periods <= 0:
            raise RuntimeError("SWMM output file contains no reporting periods.")
        if error_code != 0:
            raise RuntimeError(f"SWMM output reports simulation error code {error_code}.")
        if flow_units not in FLOW_TO_CMS:
            raise RuntimeError(f"Unsupported SWMM flow-unit code: {flow_units}")

        all_names = _read_names(
            stream,
            id_pos,
            n_subcatch + n_nodes + n_links + n_pollutants,
        )
        subcatchment_names = all_names[:n_subcatch]

        variable_offset = obj_prop_pos + (
            (n_subcatch + 2) + (3 * n_nodes + 4) + (5 * n_links + 6)
        ) * 4
        stream.seek(variable_offset)
        subcatchment_variables = _read_i32(stream)
        stream.seek(subcatchment_variables * 4, 1)
        node_variables = _read_i32(stream)
        stream.seek(node_variables * 4, 1)
        link_variables = _read_i32(stream)
        stream.seek(link_variables * 4, 1)
        system_variables = _read_i32(stream)
        if subcatchment_variables <= 4:
            raise RuntimeError("SWMM output does not contain the subcatchment runoff-rate variable.")

        stream.seek(results_pos - 12)
        start_date_ole = struct.unpack("<d", stream.read(8))[0]
        report_step_seconds = _read_i32(stream)
        bytes_per_period = 8 + 4 * (
            n_subcatch * subcatchment_variables
            + n_nodes * node_variables
            + n_links * link_variables
            + system_variables
        )

        runoff = np.empty((periods, n_subcatch), dtype="float32")
        dates = []
        ole_epoch = datetime(1899, 12, 30)
        conversion = FLOW_TO_CMS[flow_units]
        for period in range(periods):
            period_pos = results_pos + period * bytes_per_period
            stream.seek(period_pos)
            date_ole = struct.unpack("<d", stream.read(8))[0]
            dates.append(ole_epoch + timedelta(days=date_ole))
            values = np.fromfile(
                stream,
                dtype="<f4",
                count=n_subcatch * subcatchment_variables,
            ).reshape(n_subcatch, subcatchment_variables)
            runoff[period] = values[:, 4] * conversion

    return SwmmRunoffSeries(
        subcatchment_names=subcatchment_names,
        datetimes=dates,
        runoff_cms=runoff,
        report_step_seconds=report_step_seconds,
    )


def hourly_runoff_volumes(series: SwmmRunoffSeries) -> dict[str, np.ndarray]:
    """Integrate reporting-step discharge to preceding hourly interval volumes."""
    hourly: dict[str, np.ndarray] = {}
    step = timedelta(seconds=series.report_step_seconds)
    for time_value, runoff in zip(series.datetimes, series.runoff_cms):
        interval_start = time_value - step
        hour = interval_start.replace(minute=0, second=0, microsecond=0)
        key = hour.strftime("%Y-%m-%d %H:%M:%S")
        if key not in hourly:
            hourly[key] = np.zeros(runoff.shape, dtype="float64")
        hourly[key] += np.asarray(runoff, dtype="float64") * series.report_step_seconds
    return hourly
