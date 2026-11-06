"""Read-only, paginated inspection of every registered workflow artifact."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from watershed_exchange.paths import safe_relative_path
from .artifact_registry import load_artifacts


def _readable_records(job_dir: Path, project: Path) -> list[dict]:
    """An ephemeral index: reading never writes to the download registry."""
    records = [dict(item, scope='project') for item in load_artifacts(job_dir)]
    known = {item['internal_path'] for item in records}
    candidates = [(path, project, 'project') for path in sorted(project.rglob('*'))
                  if path.is_file() and path.resolve().is_relative_to(project.resolve())
                  and path.relative_to(project).as_posix() not in known]
    candidates.extend((path, job_dir, 'run') for path in sorted((job_dir/'logs').rglob('*'))
                      if path.is_file() and path.resolve().is_relative_to((job_dir/'logs').resolve()))
    candidates.extend((job_dir/name, job_dir, 'run') for name in (
        'job.json', 'status.json', 'qa.json', 'heartbeat.json') if (job_dir/name).is_file())
    for path, root, scope in candidates:
        relative = path.relative_to(root).as_posix()
        key = f'{job_dir.resolve()}:{scope}:{relative}'.encode('utf-8')
        records.append({'artifact_id':'artifact-read-'+hashlib.sha256(key).hexdigest()[:20],
                        'internal_path':relative, 'scope':scope, 'title':relative,
                        'stage_id':'inputs' if relative.startswith('inputs/') else 'run_context',
                        'size_bytes':path.stat().st_size})
    return records


def readable_catalog(job_dir: Path, project: Path) -> list[dict]:
    """All files of this task, including inputs, configuration and worker logs."""
    return [{
        'artifact_id':r['artifact_id'], 'title':r.get('title') or Path(r['internal_path']).name,
        'format':Path(r['internal_path']).suffix.lstrip('.'),
        'stage_id':r.get('stage_id'), 'role':r.get('kind') or 'task_file',
        'size_bytes':r.get('size_bytes'), 'units':r.get('units'), 'read_only':True,
    } for r in _readable_records(job_dir, project)]


def inspect_artifact(job_dir: Path, project: Path, artifact_id: str, *,
                     offset: int = 0, limit: int = 30, search: str = '') -> dict:
    if not 0 <= offset <= 10_000_000 or not 1 <= limit <= 100 or len(search) > 200:
        raise ValueError('读取页码、数量或检索文本超出范围')
    record = next((r for r in _readable_records(job_dir, project) if r['artifact_id'] == artifact_id), None)
    if record is None:
        raise ValueError('该成果不属于当前运行')
    root = project if record.get('scope') == 'project' else job_dir
    path = (root / safe_relative_path(record['internal_path'])).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('成果文件不在当前运行目录中')
    result = {'artifact_id': artifact_id, 'title': path.name, 'offset': offset,
              'size_bytes': path.stat().st_size, 'read_only': True}
    suffix = path.suffix.lower()
    if suffix in {'.tif', '.tiff'}:
        import rasterio
        import numpy as np
        with rasterio.open(path) as src:
            if src.count:
                count, total, minimum, maximum = 0, 0., math.inf, -math.inf
                for _, window in src.block_windows(1):
                    data = src.read(1, window=window, masked=True)
                    values = data.data[~np.ma.getmaskarray(data) & np.isfinite(data.data)]
                    if len(values):
                        count += len(values); total += float(values.sum(dtype='float64'))
                        minimum = min(minimum, float(values.min())); maximum = max(maximum, float(values.max()))
                result.update(kind='raster', crs=str(src.crs), width=src.width, height=src.height,
                              resolution=list(src.res),
                              resolution_unit=src.crs.linear_units if src.crs and src.crs.is_projected else 'degree' if src.crs and src.crs.is_geographic else None,
                              valid_cells=count,
                              minimum=minimum if count else None, maximum=maximum if count else None,
                              mean=total/count if count else None, units=record.get('units'))
        if 'minimum_safety_factor' in path.name:
            from .result_distribution import safety_distribution
            result['spatial_distribution'] = safety_distribution(path)
    elif suffix in {'.csv', '.tsv'}:
        result.update(kind='table', rows=[])
        matched = 0
        numeric = {}
        with path.open(encoding='utf-8-sig', errors='replace', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t' if suffix == '.tsv' else ',')
            result['columns'] = reader.fieldnames or []
            for row in reader:
                if search and search.casefold() not in json.dumps(row, ensure_ascii=False).casefold():
                    continue
                if offset <= matched < offset+limit:
                    result['rows'].append(row)
                matched += 1
                for key, raw in row.items():
                    try:
                        value = float(raw)
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(value):
                        stats = numeric.setdefault(key, {'count': 0, 'minimum': value, 'maximum': value, 'sum': 0.})
                        stats['count'] += 1; stats['sum'] += value
                        stats['minimum'] = min(stats['minimum'], value); stats['maximum'] = max(stats['maximum'], value)
        result['matched_rows'] = matched
        result['next_offset'] = offset+limit if matched > offset+limit else None
        result['column_statistics'] = {k: {**v, 'mean': v['sum']/v['count']} for k, v in numeric.items()}
        if {'metric', 'value'} <= set(result['columns']):
            result['metric_values'] = {row['metric']: row['value'] for row in result['rows']}
    elif suffix in {'.shp', '.dbf'}:
        import shapefile
        base = path.with_suffix('.shp')
        if not base.resolve().is_relative_to(project.resolve()):
            raise ValueError('矢量文件不在当前运行目录中')
        encoding = path.with_suffix('.cpg').read_text(encoding='utf-8-sig').strip() if path.with_suffix('.cpg').exists() else 'utf-8'
        with shapefile.Reader(str(base), encoding=encoding, encodingErrors='replace') as reader:
            fields = [f[0] for f in reader.fields[1:]]
            records = []; matched = 0
            for row in reader.iterRecords():
                value = dict(zip(fields, row))
                if search and search.casefold() not in json.dumps(value, ensure_ascii=False).casefold():
                    continue
                if offset <= matched < offset+limit:
                    records.append(value)
                matched += 1
            result.update(kind='vector', feature_count=len(reader), geometry_type=reader.shapeTypeName,
                          columns=fields, records=records, matched_rows=matched,
                          next_offset=offset+limit if matched > offset+limit else None)
    elif suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            result.update(kind='archive', member_count=len(names), members=names[offset:offset+limit],
                          next_offset=offset+limit if len(names)>offset+limit else None)
    elif suffix == '.out':
        from .result_builder import inspect_swmm_model
        result.update(kind='swmm_binary', model=inspect_swmm_model(project),
                      note='SWMM原生二进制；水文过程可读取同目录已登记的径流CSV和运行报告')
    elif suffix in {'.png', '.jpg', '.jpeg'}:
        from PIL import Image
        with Image.open(path) as image:
            result.update(kind='image', width=image.width, height=image.height,
                          note='图像元数据；定量分析应读取对应源栅格或表格，不按图片颜色猜数值')
    elif suffix in {'.json', '.geojson'} and path.stat().st_size <= 64*1024:
        result.update(kind='json', data=json.loads(path.read_text(encoding='utf-8-sig')))
    elif suffix in {'.txt', '.log', '.md', '.json', '.geojson', '.inp', '.rpt', '.prj', '.cpg', '.yaml', '.yml'}:
        lines = []; matched = 0
        with path.open(encoding='utf-8-sig', errors='replace') as stream:
            for line in stream:
                if search and search.casefold() not in line.casefold():
                    continue
                if offset <= matched < offset+limit:
                    lines.append(line.rstrip()[:1500])
                matched += 1
        text = '\n'.join(lines)[:30000]
        result.update(kind='text', text=text, matched_lines=matched,
                      next_offset=offset+limit if matched > offset+limit else None,
                      numeric_literals=[float(x) for x in re.findall(r'(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)][:300])
    else:
        result.update(kind='binary', note='该文件可下载；未提供此格式的内容解析器，不能声称已读取其正文')
    return result
