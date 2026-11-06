from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from watershed_hydro_mcp import grass_runtime as g


def inputs(tmp_path):
    folder = tmp_path / '中文测试目录'
    folder.mkdir()
    dem, worker = folder / '地形.tif', folder / '划分.py'
    dem.write_bytes(b'original-dem-bytes')
    worker.write_text('worker contents', encoding='utf-8')
    return dem, worker, folder / '输出'


def test_unicode_paths_are_staged_without_changing_dem_or_threshold(tmp_path, monkeypatch):
    dem, worker, output = inputs(tmp_path)
    workspaces = []
    monkeypatch.setenv('PYTHONPATH', 'foreign-python')
    monkeypatch.setenv('VIRTUAL_ENV', 'foreign-venv')
    monkeypatch.setenv('PYTHONHOME', 'foreign-home')
    monkeypatch.setenv('TEMP', 'original-temp')

    def execute(command, **kwargs):
        local_dem = Path(command[2])
        local_worker, argument_dem, local_output = map(Path, command[5:8])
        workspaces.append(local_dem.parent)
        assert local_dem != dem and argument_dem == local_dem
        assert local_dem.read_bytes() == dem.read_bytes()
        assert local_worker.read_bytes() == worker.read_bytes()
        assert command[8] == '1282'
        assert kwargs['env']['TEMP'] == str(local_dem.parent)
        assert kwargs['env']['TMPDIR'] == str(local_dem.parent)
        assert not any(k in kwargs['env'] for k in ['PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONHOME'])
        assert not kwargs.get('text', False)
        for name in g.GRASS_OUTPUTS:
            (local_output / name).write_bytes(name.encode())
        return SimpleNamespace(returncode=0, stdout=b'OK', stderr=b'')

    monkeypatch.setattr(g.subprocess, 'run', execute)
    g.run_grass(Path('grass'), dem, worker, output, 1282)
    assert all((output / n).read_bytes() == n.encode() for n in g.GRASS_OUTPUTS)
    assert dem.read_bytes() == b'original-dem-bytes'
    assert all(not p.exists() for p in workspaces)
    assert g.os.environ['TEMP'] == 'original-temp'


def test_failure_keeps_decodable_native_error_and_no_success_outputs(tmp_path, monkeypatch):
    dem, worker, output = inputs(tmp_path)
    monkeypatch.setattr(g.locale, 'getencoding', lambda: 'gbk')
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=b'', stderr='无法读取栅格'.encode('gbk')))
    with pytest.raises(RuntimeError, match='failed'):
        g.run_grass(Path('grass'), dem, worker, output, 10)
    assert '无法读取栅格' in (output / 'execution.log').read_text(encoding='utf-8')
    assert not (output / 'basin.tif').exists()


def test_missing_outputs_are_not_replaced_by_previous_results(tmp_path, monkeypatch):
    dem, worker, output = inputs(tmp_path)
    output.mkdir()
    (output / 'basin.tif').write_bytes(b'previous-run')
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=b'', stderr=b''))
    with pytest.raises(RuntimeError, match='required outputs'):
        g.run_grass(Path('grass'), dem, worker, output, 10)
    assert (output / 'basin.tif').read_bytes() == b'previous-run'


def test_timeout_keeps_partial_diagnostic(tmp_path, monkeypatch):
    dem, worker, output = inputs(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 3600, output=b'partial', stderr=b'error')

    monkeypatch.setattr(g.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        g.run_grass(Path('grass'), dem, worker, output, 10)
    assert 'partial\nerror' in (output / 'execution.log').read_text(encoding='utf-8')
