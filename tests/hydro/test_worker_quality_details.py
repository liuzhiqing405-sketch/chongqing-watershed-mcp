from watershed_hydro_mcp.worker import quality_failure_details
from watershed_hydro_mcp.qa import compact_report
import json


def test_polling_status_keeps_gate_reasons_without_large_facts():
    report = {'passed': False, 'failed': ['rainfall', 'spatial'], 'gates': [
        {'name': 'rainfall', 'errors': ['全部站点降雨总量为0'], 'facts': {'huge': [1]*1000}},
        {'name': 'spatial', 'errors': ['填洼最大抬高超过门槛']}]}
    assert quality_failure_details(report) == {
        'qa_failed_gates': ['rainfall', 'spatial'],
        'qa_errors': ['全部站点降雨总量为0', '填洼最大抬高超过门槛']}


def test_no_quality_report_has_no_invented_failures():
    assert quality_failure_details(None) == {'qa_failed_gates': [], 'qa_errors': []}


def test_large_boundary_audit_does_not_overflow_qa_tool_reply():
    report = {'passed': False, 'failed': ['spatial'], 'gates': [{'errors': ['填洼超限'],
              'facts': {'boundary_filter': {'excluded_units': [
                  {'label': i, 'reason': 'independent_boundary_outlet_below_minimum_area'}
                  for i in range(2500)], 'excluded_fraction': .005}}}]}
    compact = compact_report(report)
    assert compact['passed'] is False and compact['failed'] == report['failed']
    assert compact['gates'][0]['errors'] == report['gates'][0]['errors']
    facts = compact['gates'][0]['facts']['boundary_filter']
    assert len(facts['excluded_units']) == 20 and facts['excluded_units_total'] == 2500
    assert facts['excluded_units_truncated'] and facts['excluded_fraction'] == .005
    assert len(report['gates'][0]['facts']['boundary_filter']['excluded_units']) == 2500
    assert len(json.dumps(compact)) < 10_000
