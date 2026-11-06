from copy import deepcopy

import pytest

from watershed_exchange.domain_policy import LOCAL_SCOPE, drawn_modeling_allowed, modeling_check


@pytest.mark.parametrize("diagnostic",[
    {"checked":True,"passed":False,"classification":"upstream_truncated","errors":["外部入流"]},
    {"checked":False,"passed":False,"errors":["预算不足"]},
    {"checked":False,"passed":False,"errors":["平坦区"]},
])
def test_diagnostic_limitations_allow_explicit_local_scope_only(diagnostic):
    original=deepcopy(diagnostic)
    policy=modeling_check(diagnostic,"regional_screening")
    assert drawn_modeling_allowed({"passed":True,"hydrologic_check":diagnostic,"modeling_check":policy})
    assert policy["warnings"] and policy["boundary_assumptions"]==[LOCAL_SCOPE]
    strict=modeling_check(diagnostic,"catchment_check")
    assert not strict["passed"]
    assert not drawn_modeling_allowed({"passed":True,"hydrologic_check":diagnostic,"modeling_check":strict})
    assert diagnostic==original
    assert not drawn_modeling_allowed({"passed":True,"hydrologic_check":diagnostic})


@pytest.mark.parametrize("field,value",[("version",0),("external_inflow","included"),
    ("boundary_assumptions",[]),("assessment_mode","unknown"),("passed",False)])
def test_incomplete_or_changed_contract_cannot_unlock_modeling(field,value):
    diagnostic={"checked":True,"passed":False,"errors":["外部入流"]}
    policy=modeling_check(diagnostic,"regional_screening");policy[field]=value
    assert not drawn_modeling_allowed({"passed":True,"hydrologic_check":diagnostic,"modeling_check":policy})


def test_input_failure_and_requested_mode_mismatch_remain_blocking():
    diagnostic={"checked":False,"passed":False,"blocking_input_error":True,"errors":["无效地形"]}
    policy=modeling_check(diagnostic,"regional_screening")
    assert not policy["passed"]
    diagnostic={"checked":True,"passed":False,"errors":["外部入流"]}
    policy=modeling_check(diagnostic,"regional_screening")
    assert not drawn_modeling_allowed({"passed":True,"hydrologic_check":diagnostic,"modeling_check":policy},"catchment_check")
    assert not drawn_modeling_allowed({"passed":False,"hydrologic_check":diagnostic,"modeling_check":policy})


def test_spatial_package_and_hydro_import_preserve_regional_policy(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from fixtures import build_package
    from chongqing_spatial_mcp.exchange_export import build_manifest
    from watershed_hydro_mcp.input_registry import _accept_package
    from watershed_hydro_mcp.settings import Settings
    package=build_package(tmp_path/'package-regional')
    diagnostic={'checked':True,'passed':False,'errors':['框外入流']}
    policy=modeling_check(diagnostic,'regional_screening')
    review={'selection_mode':'drawn_study_area','review_id':'basin-testregional',
            'qa':{'passed':True,'modeling_check':policy,'hydrologic_check':diagnostic}}
    manifest=build_manifest(package,review=review,
            rain_manifest={'event_start':'2024-06-01 00:00','event_end':'2024-06-01 05:00'},qa={'passed':True},
            terrain_budget={'maximum_cells': 8_000_000, 'cells': 100})
    (package/'manifest.json').write_text(manifest.model_dump_json(),encoding='utf-8')
    settings=Settings(tmp_path,tmp_path/'runs',tmp_path/'hydro',tmp_path/'native',tmp_path/'config')
    record=_accept_package(package,settings)
    assert any('drawn by the user' in item for item in record['limitations'])
    assert not any('boundaries were produced by automatic' in item for item in record['limitations'])
    assert record['provenance']['domain_policy']==policy
    assert record['provenance']['dem_processing']=={'maximum_cells': 8_000_000, 'cells': 100}
    assert record['limitations'][0]==LOCAL_SCOPE
    assert not record['provenance']['watershed_qa']['hydrologic_check']['passed']
