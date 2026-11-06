import json
import pytest

from watershed_hydro_mcp.artifact_inspection import inspect_artifact, readable_catalog


def fixture_project(tmp_path, monkeypatch, contents, name='params.csv'):
    project=tmp_path/'project'; project.mkdir()
    (project/name).write_text(contents, encoding='utf-8')
    monkeypatch.setattr('watershed_hydro_mcp.artifact_inspection.load_artifacts',
        lambda job: [{'artifact_id':'artifact-this', 'internal_path':name}])
    return project


def test_table_search_paginates_and_statistics_use_all_matching_rows(tmp_path,monkeypatch):
    project=fixture_project(tmp_path,monkeypatch,'landuse,cohesion_kpa\nforest,7\ncrop,5\nforest,9\n')
    result=inspect_artifact(tmp_path,project,'artifact-this',search='forest',limit=1)
    assert result['matched_rows']==2 and len(result['rows'])==1 and result['next_offset']==1
    assert result['column_statistics']['cohesion_kpa']['mean']==8
    page2=inspect_artifact(tmp_path,project,'artifact-this',search='forest',limit=1,offset=1)
    assert page2['rows'][0]['cohesion_kpa']=='9' and page2['next_offset'] is None


def test_reader_rejects_other_run_ids_and_registry_path_traversal(tmp_path,monkeypatch):
    project=fixture_project(tmp_path,monkeypatch,'x\n1\n')
    with pytest.raises(ValueError): inspect_artifact(tmp_path,project,'artifact-elsewhere')
    monkeypatch.setattr('watershed_hydro_mcp.artifact_inspection.load_artifacts',
        lambda job: [{'artifact_id':'artifact-this','internal_path':'../secret.txt'}])
    with pytest.raises(ValueError): inspect_artifact(tmp_path,project,'artifact-this')


def test_text_can_search_beyond_the_initial_page_without_writing(tmp_path,monkeypatch):
    project=fixture_project(tmp_path,monkeypatch,'HEADER\n'*100+'Continuity Error = 0.01\n','model.rpt')
    before=(project/'model.rpt').read_bytes()
    result=inspect_artifact(tmp_path,project,'artifact-this',search='Continuity')
    assert result['matched_lines']==1 and '0.01' in result['text']
    assert (project/'model.rpt').read_bytes()==before


def test_catalog_includes_unregistered_inputs_configuration_and_job_logs_read_only(tmp_path,monkeypatch):
    project=fixture_project(tmp_path,monkeypatch,'x\n1\n')
    (project/'inputs').mkdir()
    (project/'inputs'/'raw_rain.csv').write_text('station,mm\nA,20\n',encoding='utf-8')
    (project/'config.yaml').write_text('cohesion_kpa: 7\n',encoding='utf-8')
    (tmp_path/'logs').mkdir()
    (tmp_path/'logs'/'swmm.log').write_text('计算完成，连续性误差为0.01\n',encoding='utf-8')
    (tmp_path/'qa.json').write_text('{"passed":false}',encoding='utf-8')
    (tmp_path/'private.txt').write_text('not task metadata',encoding='utf-8')
    before={p.relative_to(tmp_path).as_posix():p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    catalog=readable_catalog(tmp_path,project)
    by_title={r['title']:r for r in catalog}
    assert {'config.yaml','inputs/raw_rain.csv','logs/swmm.log','qa.json'}<=by_title.keys()
    assert 'private.txt' not in by_title
    log=inspect_artifact(tmp_path,project,by_title['logs/swmm.log']['artifact_id'])
    assert 0.01 in log['numeric_literals']
    assert inspect_artifact(tmp_path,project,by_title['qa.json']['artifact_id'])['data']['passed'] is False
    assert catalog==readable_catalog(tmp_path,project)
    after={p.relative_to(tmp_path).as_posix():p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after
