"""Exercise startup preflight without stopping or starting project services."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell launcher')
SCRIPT = Path(__file__).resolve().parents[2]/'scripts/start_agent_latest.ps1'


def check(script=SCRIPT):
    return subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass',
                           '-File',str(script),'-CheckOnly'],capture_output=True,
                          text=True,errors='replace',timeout=30)


def listener(port):
    server=socket.socket()
    try:
        server.bind(('127.0.0.1',port))
        server.listen()
    except OSError:
        server.close()
        pytest.skip(f'Port {port} already in use; preserve existing service')
    return server


def test_web_port_conflict_does_not_stop_its_owner():
    with listener(8780):
        result=check()
        assert result.returncode != 0
        assert 'WebUI port 8780 is occupied' in result.stderr
        with socket.create_connection(('127.0.0.1',8780),timeout=2):
            pass


def test_unknown_mcp_listener_is_preserved(tmp_path):
    # No matching service registry: a live listener is never considered owned.
    root=tmp_path/'project';(root/'scripts').mkdir(parents=True)
    copied=root/'scripts/start_agent_latest.ps1';shutil.copy2(SCRIPT,copied)
    python=root/'.venv/Scripts/python.exe';python.parent.mkdir(parents=True);python.touch()
    with listener(8765):
        result=check(copied)
        assert result.returncode != 0
        assert 'no matching project service record' in result.stderr
        with socket.create_connection(('127.0.0.1',8765),timeout=2):
            pass


@pytest.mark.parametrize('status',['queued','running','cancel_requested'])
def test_active_calculation_prevents_reload_before_identity_queries(tmp_path,status):
    root=tmp_path/'project';(root/'scripts').mkdir(parents=True)
    copied=root/'scripts/start_agent_latest.ps1';shutil.copy2(SCRIPT,copied)
    # This case exercises the active-job check with a free WebUI port. Do not
    # let the developer's unrelated, running WebUI short-circuit the fixture.
    copied.write_text(copied.read_text(encoding='utf-8-sig').replace(
        'if (Get-ListenerPid 8780) {', 'if ($false) {'), encoding='utf-8-sig')
    python=root/'.venv/Scripts/python.exe';python.parent.mkdir(parents=True);python.touch()
    job=root/'runs/hydro/jobs/hydro-run-test';job.mkdir(parents=True)
    (job/'status.json').write_text(json.dumps({'status':status,'summary':'计算未结束'},ensure_ascii=False),encoding='utf-8')
    result=check(copied)
    assert result.returncode != 0
    assert 'Active work prevents service reload' in result.stderr
    assert json.loads((job/'status.json').read_text(encoding='utf-8'))['status']==status


def test_check_only_preserves_empty_project_without_prompting_for_model_key(tmp_path):
    # Listener-dependent check is covered by the real workspace preflight.
    text=SCRIPT.read_text(encoding='utf-8')
    assert text.index('if ($CheckOnly)') < text.index('Stop-Process')
    assert text.index('Active work prevents') < text.index('Stop-Process')
    assert 'CreationDate' in text
    assert 'Read-Host' not in text
    assert '$LASTEXITCODE' not in text
