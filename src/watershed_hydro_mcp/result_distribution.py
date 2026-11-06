"""Read spatial evidence from actual result rasters; never infer it from a plot."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer


def safety_distribution(path: Path) -> dict:
    info = path.stat()
    return deepcopy(_distribution(str(path.resolve()), info.st_size, info.st_mtime_ns))


@lru_cache(maxsize=16)
def _distribution(filename: str, size: int, modified: int) -> dict:
    counts = np.zeros(4, dtype=np.int64)
    sectors = np.zeros((9, 2), dtype=np.int64)
    labels = ('西北部', '北部', '东北部', '西部', '中部', '东部', '西南部', '南部', '东南部')
    with rasterio.open(filename) as src:
        if src.crs is None or not src.crs.is_projected:
            return {'available': False, 'reason': '当前结果缺少可用于面积统计的投影坐标系'}
        cell_area = abs(src.transform.determinant) * src.crs.linear_units_factor[1] ** 2 / 1e6
        to_wgs = Transformer.from_crs(src.crs, 4326, always_xy=True)
        west, south, east, north = to_wgs.transform_bounds(*src.bounds, densify_pts=21)
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)
            good = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
            rr, cc = np.nonzero(good)
            if not len(rr):
                continue
            values = data.data[good]
            bands = np.searchsorted([1., 1.25, 1.5], values, side='right')
            counts += np.bincount(bands, minlength=4)
            rows, cols = rr + window.row_off + .5, cc + window.col_off + .5
            x = src.transform.a * cols + src.transform.b * rows + src.transform.c
            y = src.transform.d * cols + src.transform.e * rows + src.transform.f
            lon, lat = to_wgs.transform(x, y)
            sx = np.clip(((lon-west)/max(east-west, 1e-12)*3).astype(int), 0, 2)
            sy = np.clip(((north-lat)/max(north-south, 1e-12)*3).astype(int), 0, 2)
            ids = sy*3+sx
            sectors[:, 0] += np.bincount(ids, minlength=9)
            sectors[:, 1] += np.bincount(ids[values < 1], minlength=9)
    total, low = int(counts.sum()), int(counts[0])
    if not total:
        return {'available': False, 'reason': '结果栅格没有有效像元'}
    band_labels = ('Fs<1', '1≤Fs<1.25', '1.25≤Fs<1.5', 'Fs≥1.5')
    regions = {label: {'valid_cells': int(n), 'below_one_cells': int(k),
                       'area_km2': int(n)*cell_area, 'below_one_area_km2': int(k)*cell_area,
                       'below_one_fraction': int(k)/int(n) if n else None,
                       'share_of_all_below_one': int(k)/low if low else 0.}
               for label, (n, k) in zip(labels, sectors) if n}
    return {'available': True, 'source_file': Path(filename).name,
            'basis': 'event_minimum_per_pixel', 'thresholds': [1., 1.25, 1.5],
            'spatial_basis': '按实际栅格经纬度包络划分方位；空间对应不表示成因',
            'valid_cells': total, 'calculation_area_km2': total*cell_area,
            'below_one_area_km2': low*cell_area, 'below_one_fraction': low/total,
            'at_or_above_one_fraction': 1-low/total,
            'bands': {label: {'cells': int(n), 'area_km2': int(n)*cell_area, 'fraction': int(n)/total}
                      for label, n in zip(band_labels, counts)},
            'sectors': regions,
            'largest_below_one_sector': max(regions, key=lambda key: regions[key]['below_one_cells']) if low else None,
            'limitation': '每个像元在事件中曾达到的最小Fs；不能称为同一时刻的失稳面积。分段仅描述Fs数值，不是风险分级。'}
