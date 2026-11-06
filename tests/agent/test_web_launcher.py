"""The local launcher must never persist or expose a DeepSeek credential."""

from pathlib import Path
import re
import runpy
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "start_agent_webui.ps1"


def test_launcher_prompts_masked_and_keeps_the_key_off_the_command_line():
    text = SCRIPT.read_text(encoding="utf-8-sig")
    assert 'Read-Host "Model API Key (masked; not saved)" -AsSecureString' in text
    assert "SLOPE_AGENT_LLM_API_KEY" in text
    assert "slope_evaluation_agent.web" in text
    assert "--config $Config" in text
    assert not re.search(r"sk-[A-Za-z0-9_-]{6,}", text)
    web_command = next(line for line in text.splitlines()
                       if "slope_evaluation_agent.web" in line)
    assert "API_KEY" not in web_command


def test_launcher_starts_mcp_before_reading_the_masked_key():
    text = SCRIPT.read_text(encoding="utf-8-sig")
    assert text.index("& $StartServices") < text.index("Read-Host")
    assert "Remove-Item Env:SLOPE_AGENT_LLM_API_KEY" in text


def test_launcher_does_not_use_last_exit_code_for_a_powershell_script():
    text = SCRIPT.read_text(encoding="utf-8-sig")
    assert "$LASTEXITCODE" not in text
    assert "if (-not $?)" in text
    assert "Test-LocalListener -Port $SpatialPort" in text
    assert "Test-LocalListener -Port $HydroPort" in text


def test_launchers_have_a_non_admin_listener_detection_path():
    launcher = SCRIPT.read_text(encoding="utf-8-sig")
    service_script = (SCRIPT.parent / "start_mcp_services.ps1").read_text(
        encoding="utf-8-sig"
    )
    assert "GetActiveTcpListeners" in launcher
    assert "GetActiveTcpListeners" in service_script
    assert "netstat -ano -p TCP" in service_script


@pytest.mark.parametrize("default,minimum,accepted", [(0, 0, True), (5, 1, False), (0, 1, False)])
def test_launcher_detects_stale_manual_map_contract(default, minimum, accepted):
    check = runpy.run_path(str(SCRIPT.parent / "check_mcp_contract.py"))["contract_problems"]
    rows = [SimpleNamespace(name="render_outlet_candidates", input_schema={
        "properties": {"max_candidates": {"default": default, "minimum": minimum},
                       "longitude": {}, "latitude": {}}})]
    assert bool(check(rows, {"render_outlet_candidates"})) == (not accepted)
    assert check(rows, {"get_hydro_capabilities"}) == []


def test_launcher_detects_stale_coordinate_map_contract():
    check = runpy.run_path(str(SCRIPT.parent / "check_mcp_contract.py"))["contract_problems"]
    rows = [SimpleNamespace(name="render_outlet_candidates", input_schema={
        "properties": {"max_candidates": {"default": 0, "minimum": 0}}})]
    assert check(rows, {"render_outlet_candidates"})


def test_launcher_requires_the_manual_map_tool():
    text = SCRIPT.read_text(encoding="utf-8-sig")
    required = text.split("$SpatialRequiredTools = @(", 1)[1].split(")", 1)[0]
    assert '"render_outlet_candidates"' in required


@pytest.mark.parametrize('mode,accepted',[
    ({},False),({'default':'regional_screening'},False),
    ({'default':'catchment_check','enum':['regional_screening','catchment_check']},False),
    ({'default':'regional_screening','enum':['regional_screening','catchment_check']},True)])
def test_launcher_detects_old_drawn_tool_even_when_name_exists(mode,accepted):
    check=runpy.run_path(str(SCRIPT.parent/'check_mcp_contract.py'))['contract_problems']
    rows=[SimpleNamespace(name='create_drawn_study_area',input_schema={'properties':{'assessment_mode':mode}})]
    assert bool(check(rows,{'create_drawn_study_area'})) == (not accepted)
