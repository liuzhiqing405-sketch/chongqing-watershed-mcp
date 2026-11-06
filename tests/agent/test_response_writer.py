"""Phase 8 acceptance: the reply layer.

Two properties matter here. Every number a reply shows must come from evidence,
and no reply may assert something the scope forbids — however the wording was
assembled.
"""

from __future__ import annotations

import pytest

from slope_evaluation_agent import DISCLAIMER_ZH, SCOPE_STATEMENT
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.response_writer import (
    FORBIDDEN_PHRASES,
    describe_cancellation,
    describe_capabilities,
    describe_catalog,
    describe_confirmation,
    describe_data_check,
    describe_import,
    describe_input_validation,
    describe_model_inputs,
    describe_outlet_candidates,
    describe_plan,
    describe_rainfall_package,
    describe_review,
    describe_run_started,
    describe_run_status,
    describe_scope,
    describe_unknown,
    describe_visualization_ready,
    find_forbidden,
    sanitize,
    scrub_paths,
    stamp_limitations,
)
from slope_evaluation_agent.scientific_policy import numeric_audit
from slope_evaluation_agent.state_machine import WorkflowState


def a_session() -> AgentSession:
    return AgentSession.model_validate(
        {"session_id": "session-abcdef123456", "state": "new"})


def test_failed_quality_reply_explains_each_gate_without_claiming_success():
    reply = describe_run_status({'status': 'qa_failed', 'run_id': 'hydro-run-test',
                                'completed_stages': ['qa'],
                                'qa_errors': ['全部站点降雨总量为0', '填洼最大抬高超过门槛']})
    assert '全部站点降雨总量为0' in reply and '填洼最大抬高超过门槛' in reply
    assert '不能作为稳定性结论' in reply


class TestForbiddenPhraseAudit:
    def test_no_template_asserts_a_forbidden_claim(self):
        """Whatever the shape of the payload, a template may not assert these."""
        samples = [
            describe_scope(),
            describe_unknown(a_session(), "随便"),
            describe_data_check({"ok": True, "data": {"rainfall_years": [2020, 2021]}}),
            describe_outlet_candidates({"data": {"candidates": [], "message": "无"}}, "大宁河"),
            describe_review({"review_id": "basin-abcdef123456",
                             "qa": {"passed": True, "area_km2": 2.15},
                             "outlet": {"snapped_longitude": 109.5, "snapped_latitude": 31.4}}),
            describe_confirmation({"review_id": "basin-abcdef123456"}, confirmed=True),
            describe_rainfall_package({"package_id": "rain-abcdef123456",
                                       "selection_mode": "historical_max",
                                       "stations": [{"station_key": "VS01"}],
                                       "statistics_years": [2015, 2016],
                                       "timestamp_count": 24,
                                       "event_start": "2024-07-30 00:00:00",
                                       "event_end": "2024-07-30 23:00:00"}),
            describe_model_inputs({"package_id": "package-abcdef123456",
                                   "exchange": {"files": ["dem.tif"]}}),
            describe_import({"input_id": "hydro-input-abcdef123456"}),
            describe_input_validation({"input_id": "x", "ok": True}),
            describe_plan({"stages": [{"stage_id": "sync"}], "parameter_overrides": {}},
                          preset="medium"),
            describe_run_started({"run_id": "hydro-run-abcdef123456", "status": "queued",
                                  "stages": ["sync"]}),
            describe_run_status({"run_id": "hydro-run-abcdef123456", "status": "succeeded",
                                 "completed_stages": ["sync"]}),
            describe_cancellation({"run_id": "hydro-run-abcdef123456", "status": "cancelled"}),
            describe_visualization_ready({"title": "图", "renderer_version": "1.0",
                                          "sources": []}),
            describe_catalog([{"kind": "safety_minimum_factor", "stage_id": "safety"}]),
            describe_capabilities({"data": {"service": "x", "engine": {"ok": True},
                                            "modules": {"ok": True}, "stages": [],
                                            "parameter_presets": ["medium"],
                                            "registered_inputs": 0, "limitations": []}}),
        ]
        for text in samples:
            assert find_forbidden(text) == [], text[:120]

    def test_the_scope_statement_matches_the_plan(self):
        assert SCOPE_STATEMENT in describe_scope()
        assert "不是工程级边坡安全鉴定" in describe_scope()

    def test_stamp_limitations_always_carries_the_three_standing_caveats(self):
        stamped = " ".join(stamp_limitations([]))
        assert "初步筛查与诊断" in stamped
        assert "虚拟雨站" in stamped
        assert "路由流量过程" in stamped


class TestNumbersComeFromThePayload:
    def test_candidate_numbers_are_the_servers_numbers(self):
        payload = {"data": {"candidates": [
            {"longitude": 109.5123, "latitude": 31.4456, "elevation_m": 210.4,
             "dem_edge_distance_km": 30.2, "matched_river": "大宁河"}],
            "matched_names": ["大宁河"]}}
        text = describe_outlet_candidates(payload, "大宁河")
        assert "109.5123" in text
        assert "210.4" in text
        assert numeric_audit(text, payload) == []

    def test_review_numbers_are_the_servers_numbers(self):
        review = {"review_id": "basin-abcdef123456",
                  "outlet": {"snapped_longitude": 108.0274415, "snapped_latitude": 30.4449601,
                             "shift_m": 153.0},
                  "qa": {"passed": True, "area_km2": 2.151, "selected_cells": 2389}}
        text = describe_review(review)
        assert "2.151" in text
        assert "2,389" in text
        assert numeric_audit(text, review) == []

    def test_rainfall_numbers_are_the_servers_numbers(self):
        package = {"package_id": "rain-abcdef123456", "selection_mode": "historical_max",
                   "stations": [{"station_key": "VS01"}], "statistics_years": [2015, 2026],
                   "timestamp_count": 24, "event_start": "2024-07-30 00:00:00",
                   "event_end": "2024-07-30 23:00:00",
                   "selected_day_basin_station_mean_total_mm": 79.29}
        text = describe_rainfall_package(package)
        assert "79.29" in text
        assert numeric_audit(text, package) == []

    def test_no_template_invents_a_number_when_the_payload_is_empty(self):
        text = describe_rainfall_package({"package_id": "rain-abcdef123456"})
        assert "None" not in text
        assert "nan" not in text.lower()


class TestHistoricalMaximumWording:
    def test_a_sample_maximum_is_named_as_such(self):
        text = describe_rainfall_package({
            "package_id": "rain-abcdef123456", "selection_mode": "historical_max",
            "stations": [], "statistics_years": [2015, 2026], "timestamp_count": 24,
            "event_start": "2024-07-30 00:00:00", "event_end": "2024-07-30 23:00:00",
        })
        assert "样本最大" in text
        assert "不是重现期设计暴雨" in text
        assert "2015–2026" in text

    def test_virtual_stations_are_not_called_measured(self):
        text = describe_rainfall_package({"package_id": "rain-abcdef123456",
                                          "stations": [], "selection_mode": "historical_max"})
        assert "不是地面实测雨量站" in text
        assert find_forbidden(text) == []


class TestRunStatusBranches:
    def test_succeeded_says_it_may_be_assessed(self):
        text = describe_run_status({"run_id": "hydro-run-abcdef123456", "status": "succeeded",
                                    "completed_stages": ["sync"]})
        assert "质量门槛" in text

    def test_qa_failed_says_diagnostic_only(self):
        text = describe_run_status({"run_id": "hydro-run-abcdef123456", "status": "qa_failed",
                                    "completed_stages": ["sync"]})
        assert "只能用于诊断" in text
        assert "不能作为稳定性结论" in text

    def test_failed_names_the_reason(self):
        text = describe_run_status({"run_id": "hydro-run-abcdef123456", "status": "failed",
                                    "summary": "阶段失败：运行SWMM"})
        assert "运行SWMM" in text

    def test_cancelled_offers_a_retry(self):
        text = describe_run_status({"run_id": "hydro-run-abcdef123456",
                                    "status": "cancelled", "completed_stages": []})
        assert "重新" in text


class TestPathScrubbing:
    def test_a_windows_path_is_replaced(self):
        path = "C:" + chr(92) + "runs" + chr(92) + "jobs" + chr(92) + "x.json"
        assert "C:" not in scrub_paths(f"缺失：{path}")

    def test_a_posix_path_is_replaced(self):
        assert "/home/user/secret" not in scrub_paths("读到 /home/user/secret 失败")

    def test_a_plain_message_is_untouched(self):
        assert scrub_paths("输入校验未通过") == "输入校验未通过"


class TestGuardReplies:
    def test_data_check_failure_lists_the_reasons(self):
        text = describe_data_check({"ok": False, "data": {"errors": ["缺少 DEM"]}})
        assert "缺少 DEM" in text
        assert "未通过" in text

    def test_validation_failure_lists_the_errors(self):
        text = describe_input_validation({"input_id": "x", "ok": False,
                                          "errors": ["缺少文件：landuse.shp"]})
        assert "landuse.shp" in text
        assert "重新导入" in text

    def test_unknown_intent_reports_what_is_pending(self):
        session = AgentSession.model_validate(
            {"session_id": "session-abcdef123456",
             "state": WorkflowState.AWAITING_WATERSHED_CONFIRMATION.value})
        text = describe_unknown(session, "随便说点什么")
        assert "确认" in text or "拒绝" in text


class TestScopeGuard:
    def test_the_disclaimer_is_present(self):
        assert DISCLAIMER_ZH in describe_scope()

    def test_the_disclaimer_forbids_the_right_things(self):
        for phrase in ("工程级边坡安全鉴定", "灾害预报", "泥石流易发性"):
            assert phrase in DISCLAIMER_ZH

    def test_forbidden_phrases_include_the_debris_flow_claim(self):
        assert "泥石流易发性" in FORBIDDEN_PHRASES


class TestReviewTextsSurviveTheNumericAudit:
    """A boundary reply may only quote numbers the review record carries.

    This is the trap the adaptive-window work walked into: the reply names the
    window, the grid and the gate thresholds, and a template that quotes a
    threshold the record does not hold makes the *whole* reply fail the audit —
    the user then sees "内部一致性检查未通过" instead of the boundary they asked
    for.
    """

    @staticmethod
    def review(**overrides) -> dict:
        record = {
            "review_id": "basin-abcdef123456",
            "outlet": {"snapped_longitude": 107.11, "snapped_latitude": 29.81, "shift_m": 12.5},
            "qa": {
                "passed": False,
                "errors": ["流域触及处理窗口边界（north 方向；已自适应尝试 5 次，"
                           "窗口 100 km，栅格 71 m），范围可能不完整",
                           "处理窗口已到上限（5 次尝试 / 100 km 半宽）；该流域的上游流程"
                           "超出本工具一次能覆盖的范围，请改选更靠上游、汇水面积更小的出水口（或分段分析）"],
                "notes": ["流域边缘与 DEM 无数据区接触 1173 个栅格单元（占流域 0.19%），"
                          "低于截断判定阈值 0.50%，仅作提示"],
                "area_km2": 3177.0191,
                "selected_cells": 630123,
                "touches_processing_window": True,
                "touches_dem_nodata": True,
                "touched_sides": ["north"],
                "nodata_contact_cells": 1173,
                "nodata_contact_fraction": 0.001862,
                "minimum_area_km2": 1.8,
                "minimum_nodata_contact_fraction": 0.005,
            },
            "processing": {
                "method": "Priority-Flood + D8",
                "resolution_m": 71.0,
                "maximum_processing_cells": 8000000,
                "window_km": 100.0,
                "window_width_km": 200.0,
                "window_height_km": 200.0,
                "window_attempts": 5,
                "window_clamped_sides": [],
                "adaptive_window": True,
                "native_cell_size_m": 30.0,
                "grid_coarsened": True,
            },
        }
        record.update(overrides)
        return record

    def test_the_failure_reply_audits_clean(self):
        record = self.review()
        assert numeric_audit(describe_review(record), {"review": record}) == []

    def test_the_passing_reply_audits_clean_when_the_grid_was_coarsened(self):
        record = self.review()
        record["qa"] = {**record["qa"], "passed": True, "errors": [], "notes": [],
                        "touched_sides": [], "touches_processing_window": False,
                        "touches_dem_nodata": False}
        record["processing"] = {**record["processing"], "window_km": 30.0,
                                "resolution_m": 32.0, "window_attempts": 2}
        text = describe_review(record)
        assert numeric_audit(text, {"review": record}) == []
        assert "32" in text and "30" in text          # the grid it was read at
        assert "原生" in text                          # and what it was coarsened from

    def test_a_basin_below_the_minimum_area_audits_clean(self):
        record = self.review()
        record["qa"] = {**record["qa"], "area_km2": 0.85,
                        "errors": ["流域面积0.850 km2，小于当前1.8 km2下限"]}
        assert numeric_audit(describe_review(record), {"review": record}) == []

    def test_a_boundary_graze_is_a_note_not_a_refusal(self):
        record = self.review()
        record["qa"] = {**record["qa"], "passed": True,
                        "errors": [], "touched_sides": [], "touches_processing_window": False}
        text = describe_review(record)
        assert "提示：" in text
        assert "空间质量检查未通过" not in text
