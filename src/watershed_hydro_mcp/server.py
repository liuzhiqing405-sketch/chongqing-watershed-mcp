"""The watershed hydrology and slope-stability MCP server.

This service is independent of the spatial MCP: it never imports it, never
starts it and never reads its run state. Inputs arrive either as a Model Input
Package copied into an inbox or as files registered from an inbox directory,
and every run works inside its own private project directory.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Image, MCPServer

from watershed_exchange.paths import PathSafetyError

from . import __version__, planner
from .artifact_registry import load_artifacts
from .capabilities import capabilities
from .cleanup import cleanup_hydro_products as clean_products
from .export_service import ExportService
from .input_registry import (
    InputRegistrationError,
    import_package,
    list_inputs,
    load_record,
    register_inputs,
    validate_input,
)
from .job_manager import STORE
from .models import HydroRunRequest, RegisterInputRequest
from .qa import read_gates, run_gates, compact_report
from .result_builder import (
    inspect_swmm_model as inspect_swmm_model_result,
    read_artifact_text,
    summarize,
)
from .settings import SETTINGS
from .state import envelope, failure, read_json
from .visualization import catalog as visualization_catalog
from .visualization.service import SERVICE as VISUALIZATIONS


mcp = MCPServer(
    "watershed-hydro-stability",
    version=__version__,
    instructions=(
        "中小流域水文与边坡稳定性计算服务，独立于数据与空间处理MCP。"
        "输入必须来自 Model Input Package v1 或注册的合格数据，导入后复制到本服务的受管目录。"
        "后台任务的返回状态中，qa_failed 表示计算完成但质量门槛未通过，不得当作成功结论。"
        "当前模型为各子汇水区独立出水口配置：各子汇水区径流之和不是选定流域出口的路由流量过程。"
    ),
)


#: Projects registry records onto the fields a caller may see, and turns them
#: into downloadable descriptors. Never exposes an internal path.
EXPORTS = ExportService(SETTINGS)


def _guard(call):
    """Run a tool body, converting expected failures into the response envelope."""
    try:
        with STORE._lock, VISUALIZATIONS._lock:
            return call()
    except (PathSafetyError, InputRegistrationError, planner.PlanningError) as exc:
        return failure(str(exc))
    except FileNotFoundError as exc:
        return failure(f"缺少文件：{exc}")
    except ValueError as exc:
        return failure(str(exc))
    except RuntimeError as exc:
        return failure(str(exc))


# --------------------------------------------------------------- capabilities


@mcp.tool(structured_output=True)
def get_hydro_capabilities() -> dict[str, Any]:
    """说明本服务的能力、阶段白名单、参数预设、质量门槛和当前可用性。"""
    return _guard(lambda: envelope(
        ok=True, status="ready",
        summary="已报告服务能力与当前可用性",
        facts={
            "engine_ok": capabilities()["engine"]["ok"],
            "modules_ok": capabilities()["modules"]["ok"],
            "registered_inputs": capabilities()["registered_inputs"],
        },
        data=capabilities(),
        limitations=capabilities()["limitations"],
    ))


# ---------------------------------------------------------------------- input


@mcp.tool(structured_output=True)
def import_hydro_input_package(package: str) -> dict[str, Any]:
    """从收件目录导入 Model Input Package v1（包编号或 .zip 文件名）。"""
    def call() -> dict[str, Any]:
        record = import_package(package)
        return envelope(
            ok=True, status="imported",
            summary=f"已导入输入包，复制到受管目录：{record['input_id']}",
            facts={"input_id": record["input_id"],
                   "source_package_id": record.get("source_package_id"),
                   "event": record.get("event")},
            warnings=(record.get("verification") or {}).get("warnings", []),
            limitations=record.get("limitations", []),
            next_actions=["validate_hydro_input", "plan_hydro_run", "start_hydro_run"],
            data=record,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def register_hydro_inputs(request: RegisterInputRequest) -> dict[str, Any]:
    """注册其他来源的DEM、土地利用、水系、边界和降雨；文件必须位于收件目录内。"""
    def call() -> dict[str, Any]:
        record = register_inputs(request)
        return envelope(
            ok=True, status="registered",
            summary=f"已注册并复制输入：{record['input_id']}",
            facts={"input_id": record["input_id"], "source_root": record["source_root"]},
            limitations=record.get("limitations", []),
            next_actions=["validate_hydro_input", "plan_hydro_run"],
            data=record,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def validate_hydro_input(input_id: str) -> dict[str, Any]:
    """复检已登记输入的完整性、哈希、空间与时间一致性。"""
    def call() -> dict[str, Any]:
        report = validate_input(input_id)
        return envelope(
            ok=report["ok"], status="validated" if report["ok"] else "invalid",
            summary="输入校验通过" if report["ok"] else "输入校验未通过：" + "；".join(report["errors"][:3]),
            facts={"input_id": input_id, "errors": report["errors"], "warnings": report["warnings"]},
            warnings=report["warnings"],
            limitations=report.get("limitations", []),
            next_actions=["plan_hydro_run"] if report["ok"] else ["重新导入或注册输入"],
            data=report,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def list_hydro_inputs() -> dict[str, Any]:
    """列出本服务已导入或注册的输入。"""
    def call() -> dict[str, Any]:
        rows = list_inputs()
        return envelope(
            ok=True, status="listed",
            summary=f"共 {len(rows)} 个受管输入",
            facts={"count": len(rows)},
            data={"inputs": rows},
        )
    return _guard(call)


# ------------------------------------------------------------ plan and execute


@mcp.tool(structured_output=True)
def plan_hydro_run(request: HydroRunRequest) -> dict[str, Any]:
    """生成运行计划：将要执行的阶段、事件窗口、参数和输出位置；不启动计算。"""
    def call() -> dict[str, Any]:
        record = load_record(request.input_id)
        validation = validate_input(request.input_id)
        if not validation["ok"]:
            return failure(
                "输入校验未通过，无法生成可执行计划：" + "；".join(validation["errors"][:3]),
                status="invalid",
                facts={"input_id": request.input_id, "errors": validation["errors"]},
                warnings=validation["warnings"],
                data={"validation": validation},
            )
        stages = planner.stages_for(request.mode, request.through)
        terrain = planner.terrain_preflight(record)
        presets = planner.load_presets()
        engine = planner.check_engine()
        overrides = request.parameter_overrides.present()
        bounds = presets.get("bounds") or {}
        rejected = {}
        for key, value in overrides.items():
            if key in bounds:
                low, high = bounds[key]
                if not (low <= float(value) <= high):
                    rejected[key] = f"{value} 超出 [{low}, {high}]"
        if rejected:
            return failure("参数超出允许范围：" + "；".join(f"{k} {v}" for k, v in rejected.items()),
                           status="rejected")
        warnings = []
        if not engine["ok"]:
            warnings.append("SWMM 引擎文件不完整，含建模或运行的阶段会失败")
        return envelope(
            ok=True, status="planned",
            summary=f"计划执行 {len(stages)} 个阶段：{', '.join(s.stage_id for s in stages)}",
            facts={
                "input_id": request.input_id,
                "mode": request.mode,
                "through": request.through,
                "stages": [s.stage_id for s in stages],
                "engine_present": engine["ok"],
                "input_validated": True,
                "dem_preflight": terrain,
            },
            warnings=warnings,
            limitations=["计划不代表已执行；输入尚未复制到运行目录。"],
            next_actions=["start_hydro_run"],
            data={
                "input": {"input_id": record["input_id"], "origin": record.get("origin"),
                          "event": record.get("event"), "study_area": record.get("study_area")},
                "parameter_preset": request.parameter_preset,
                "parameter_overrides": overrides,
                "parameter_bounds": bounds,
                "stages": [{"stage_id": s.stage_id, "title": s.title} for s in stages],
                "engine": engine,
            },
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def start_hydro_run(request: HydroRunRequest) -> dict[str, Any]:
    """提交后台运行；立即返回任务编号，用 get_hydro_run_status 查询进度。"""
    def call() -> dict[str, Any]:
        status = STORE.start(request)
        return envelope(
            ok=True, status=status["status"],
            summary=f"任务已提交：{status['run_id']}",
            facts={"run_id": status["run_id"], "stages": status["stages"],
                   "mode": status["mode"]},
            next_actions=["get_hydro_run_status", "cancel_hydro_run"],
            data=status,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def get_hydro_run_status(run_id: str) -> dict[str, Any]:
    """查询后台任务的状态、已完成阶段和最近日志位置。"""
    def call() -> dict[str, Any]:
        status = STORE.status(run_id)
        state = status.get("status")
        return envelope(
            ok=state in {"succeeded", "qa_failed", "running", "queued"},
            status=state,
            summary=str(status.get("summary") or ""),
            facts={"run_id": run_id, "current_stage": status.get("current_stage"),
                   "completed_stages": status.get("completed_stages", []),
                   "error": status.get("error")},
            limitations=(
                ["质量门槛未通过，结果不得作为已验证结论使用"]
                if state == "qa_failed" else []
            ),
            next_actions=(
                ["get_hydro_run_result"] if state in {"succeeded", "qa_failed"}
                else ["cancel_hydro_run"] if state in {"running", "queued"} else []
            ),
            data=status,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def cancel_hydro_run(run_id: str) -> dict[str, Any]:
    """取消排队或运行中的任务；已完成的任务不会被改动。"""
    def call() -> dict[str, Any]:
        status = STORE.cancel(run_id)
        return envelope(
            ok=True, status=status["status"],
            summary=f"任务状态：{status['status']}",
            facts={"run_id": run_id, "completed_stages": status.get("completed_stages", [])},
            data=status,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def get_hydro_run_result(run_id: str) -> dict[str, Any]:
    """汇总任务成果：SWMM 模型与连续性、安全系数摘要、降雨摘要与质量门槛结果。"""
    def call() -> dict[str, Any]:
        status = STORE.status(run_id)
        job_dir = SETTINGS.job_dir(run_id)
        project = job_dir / "project"
        summary = summarize(job_dir, project)
        state = status.get("status")
        return envelope(
            ok=state == "succeeded",
            status=state,
            summary=(
                "任务成功，质量门槛通过" if state == "succeeded"
                else "计算完成但质量门槛未通过" if state == "qa_failed"
                else f"任务状态：{state}"
            ),
            facts={
                "run_id": run_id,
                "artifact_count": summary["artifact_count"],
                "completed_stages": status.get("completed_stages", []),
                "qa_passed": (summary.get("quality") or {}).get("passed"),
            },
            limitations=summary["limitations"],
            next_actions=["list_hydro_artifacts", "list_hydro_visualization_options"],
            data={"status": status, **summary},
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def recheck_hydro_qa(run_id: str) -> dict[str, Any]:
    """对已有任务的现存成果重新执行质量检查，不重新计算。"""
    def call() -> dict[str, Any]:
        status = STORE.status(run_id)
        if status.get("status") in {"queued", "running"}:
            return failure("任务仍在执行，无法重检质量门槛", status=status.get("status"))
        job_dir = SETTINGS.job_dir(run_id)
        project = job_dir / "project"
        report = run_gates(project, list(status.get("completed_stages") or []))
        STORE.record_quality_report(run_id, report)
        return envelope(
            ok=bool(report["passed"]), status="qa_passed" if report["passed"] else "qa_failed",
            summary="质量门槛通过" if report["passed"] else f"未通过门槛：{report['failed']}",
            facts={"run_id": run_id, "failed": report["failed"]},
            limitations=[] if report["passed"] else ["结果不得作为已验证结论使用"],
            data=compact_report(report),
        )
    return _guard(call)


# ------------------------------------------------------------------ artifacts


@mcp.tool(structured_output=True)
def inspect_swmm_model(run_id: str) -> dict[str, Any]:
    """检查SWMM模型结构：子汇水区、雨量计、节点与出水口、面积、不透水率、模拟时间、连续性误差与图件。"""
    def call() -> dict[str, Any]:
        STORE.status(run_id)
        project = SETTINGS.job_dir(run_id) / "project"
        model = inspect_swmm_model_result(project)
        return envelope(
            ok=bool(model.get("available")),
            status="available" if model.get("available") else "unavailable",
            summary=(
                f"子汇水区 {model.get('subcatchment_count')} 个，出水口 {model.get('outfall_count')} 个"
                if model.get("available") else str(model.get("reason"))
            ),
            facts={"run_id": run_id, "subcatchments": model.get("subcatchment_count"),
                   "outfalls": model.get("outfall_count"),
                   "routing_network_present": model.get("routing_network_present")},
            limitations=[model["limitation"]] if model.get("limitation") else [],
            data=model,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def list_hydro_artifacts(run_id: str, stage_id: str | None = None) -> dict[str, Any]:
    """列出任务登记的成果；可用 stage_id 过滤。仅使用返回的 artifact_id 引用成果。"""
    def call() -> dict[str, Any]:
        STORE.status(run_id)
        records = load_artifacts(SETTINGS.job_dir(run_id))
        if stage_id:
            records = [item for item in records if item.get("stage_id") == stage_id]
        # Publish descriptors, not registry records. Registry records contain an
        # internal relative path; callers need the opaque id and original file
        # metadata only, and downloads resolve that id inside this service.
        records = [EXPORTS.descriptor(run_id, item) for item in records]
        return envelope(
            ok=True, status="listed",
            summary=f"共 {len(records)} 项成果",
            facts={"run_id": run_id, "count": len(records), "stage_id": stage_id},
            next_actions=["read_hydro_text_artifact", "create_hydro_visualization"],
            data={"artifacts": records},
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def list_hydro_readable_artifacts(run_id: str) -> dict[str, Any]:
    """只读列举当前任务的全部产物、输入、配置、检查及阶段日志，不修改成果登记。"""
    def call():
        STORE.status(run_id)
        from .artifact_inspection import readable_catalog
        job_dir = SETTINGS.job_dir(run_id)
        return envelope(ok=True, status='listed', summary='已列出任务可读资料',
                        data={'artifacts':readable_catalog(job_dir, job_dir/'project')})
    return _guard(call)


@mcp.tool(structured_output=True)
def inspect_hydro_artifact(run_id: str, artifact_id: str, offset: int = 0,
                           limit: int = 30, search: str = '') -> dict[str, Any]:
    """只读查阅当前任务目录给出的成果、输入、配置及日志；支持分页检索。"""
    def call():
        STORE.status(run_id)
        from .artifact_inspection import inspect_artifact
        job_dir = SETTINGS.job_dir(run_id)
        data = inspect_artifact(job_dir, job_dir/'project', artifact_id,
                                offset=offset, limit=limit, search=search)
        return envelope(ok=True, status='read', summary='已读取任务成果', data=data)
    return _guard(call)


@mcp.tool(structured_output=True)
def read_hydro_text_artifact(run_id: str, artifact_id: str,
                             limit_bytes: int = 20 * 1024) -> dict[str, Any]:
    """读取文本类成果（日志、CSV、报告），默认最多返回 20 KB。"""
    def call() -> dict[str, Any]:
        if not 1024 <= limit_bytes <= 64 * 1024:
            return failure("limit_bytes 必须在 1024 到 65536 之间")
        STORE.status(run_id)
        project = SETTINGS.job_dir(run_id) / "project"
        payload = read_artifact_text(SETTINGS.job_dir(run_id), project, artifact_id, limit_bytes)
        warnings = ["内容已截断" ] if payload.get("truncated") else []
        return envelope(
            ok=True, status="read",
            summary=f"已读取 {payload['returned_bytes']} 字节"
                    + ("（已截断）" if payload.get("truncated") else ""),
            facts={"artifact_id": artifact_id, "size_bytes": payload["size_bytes"],
                   "truncated": payload["truncated"]},
            warnings=warnings,
            data=payload,
        )
    return _guard(call)


# ------------------------------------------------------------- visualizations


@mcp.tool(structured_output=True)
def list_hydro_visualization_options(run_id: str | None = None,
                                     stage_id: str | None = None) -> dict[str, Any]:
    """列出可用的可视化类型；客户端只能选择类型，不能提交绘图代码。"""
    def call() -> dict[str, Any]:
        options = visualization_catalog.catalog()
        if stage_id:
            options = [item for item in options if item["stage_id"] == stage_id]
        existing = VISUALIZATIONS.list_for_run(run_id) if run_id else []
        return envelope(
            ok=True, status="listed",
            summary=f"共 {len(options)} 种可视化",
            facts={"count": len(options), "run_id": run_id, "stage_id": stage_id},
            data={"options": options, "existing": existing},
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def create_hydro_visualization(run_id: str, kind: str) -> dict[str, Any]:
    """提交后台渲染任务；立即返回可视化编号。"""
    def call() -> dict[str, Any]:
        record = VISUALIZATIONS.create(run_id, kind)
        return envelope(
            ok=True, status=record["status"],
            summary=f"已提交可视化任务：{record['visualization_id']}（{record['title']}）",
            facts={"visualization_id": record["visualization_id"], "kind": kind,
                   "stage_id": record["stage_id"]},
            next_actions=["get_hydro_visualization_status", "get_hydro_visualization"],
            data=record,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def get_hydro_visualization_status(visualization_id: str) -> dict[str, Any]:
    """查询可视化渲染任务的状态与源文件哈希。"""
    def call() -> dict[str, Any]:
        record = VISUALIZATIONS.status(visualization_id)
        return envelope(
            ok=record.get("status") in {"queued", "running", "succeeded"},
            status=str(record.get("status")),
            summary=str(record.get("error") or record.get("title") or ""),
            facts={"visualization_id": visualization_id,
                   "sources": len(record.get("sources") or []),
                   "qa_failed": bool(record.get("qa_failed"))},
            limitations=record.get("limitations", []),
            next_actions=["get_hydro_visualization"] if record.get("status") == "succeeded" else [],
            data=record,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def prepare_artifact_export(run_id: str, artifact_id: str, format: str) -> dict[str, Any]:
    """按白名单把一个已登记成果转换成便于查看的派生件（目前支持 xlsx）。

    原始 GeoTIFF/CSV 不需要转换，直接按 artifact_id 下载即可。派生件同样进入
    ArtifactRegistry：记录来源成果编号、转换器与版本、来源哈希，因此“派生件”
    与“计算结果”不会被混为一谈。相同来源与转换器版本会复用已有结果。
    """
    def call() -> dict[str, Any]:
        STORE.status(run_id)
        result = EXPORTS.prepare_export(run_id, artifact_id, format)
        artifact = result["artifact"]
        return envelope(
            ok=True, status="exported",
            summary=(f"复用已有 {artifact['format']} 派生件" if result["reused"]
                     else f"已生成 {artifact['format']} 派生件"),
            facts={"run_id": run_id, "source_artifact_id": artifact_id,
                   "artifact_id": artifact["artifact_id"], "format": artifact["format"],
                   "reused": result["reused"], "size_bytes": artifact.get("size_bytes")},
            limitations=["派生件由原始成果转换而来；引用数据时请以原始成果为准"],
            next_actions=["get_artifact_download"],
            data=result,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def prepare_swmm_model_bundle(run_id: str) -> dict[str, Any]:
    """把本次运行登记的 SWMM 建模成果打包成一个可复核、可复算的 ZIP。

    包内是固定目录结构：`model/model.inp`、`results/model.rpt`、`results/model.out`、
    `tables/*`（参数/降雨审查表、雨量站权重）、`forcing/rainfall_hourly.csv`、
    `spatial/subcatchments.*`，外加 `README.md`、`manifest.json`、`checksums.sha256`、
    `qa/qa.json` 与 `metadata/software_versions.json`。每个成员都来自**本运行**登记的
    成果（按登记表解析，不扫描目录），并逐文件核对摘要。

    必需文件（INP、RPT）缺失时仍会打包，但包名带 `INCOMPLETE_`、状态为不完整，
    并明确列出缺什么：这样的包只能用于诊断，不是完整模型包。
    """
    def call() -> dict[str, Any]:
        STORE.status(run_id)
        result = EXPORTS.prepare_swmm_bundle(run_id)
        artifact = result["artifact"]
        completeness = "完整" if result["complete"] else "不完整，仅供诊断"
        return envelope(
            ok=True, status="bundled",
            summary=(f"复用已有 SWMM 模型包（{completeness}）" if result["reused"]
                     else f"已生成 SWMM 模型包（{completeness}）"),
            facts={"run_id": run_id, "artifact_id": artifact["artifact_id"],
                   "complete": result["complete"],
                   "missing_required": result["missing_required"],
                   "members": len(result.get("members") or []),
                   "reused": result["reused"],
                   "size_bytes": artifact.get("size_bytes"),
                   "sha256": artifact.get("sha256")},
            warnings=result.get("warnings", []),
            limitations=["包内文件由本次运行登记成果直接打包，未做任何数值转换；"
                         "缺失必需文件时该包不能作为完整模型包使用",
                         "结果读取器目前只解析子汇水区径流变量，未导出节点与管段时间序列"],
            next_actions=["list_hydro_artifacts"],
            data=result,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def locate_artifact(run_id: str, artifact_id: str) -> dict[str, Any]:
    """返回一个已登记成果在本机的受控定位信息（路径、摘要、大小、媒体类型）。

    **仅供同机受信客户端内部使用。** 返回的 `internal_path` 是本机绝对路径，用途只有
    一个：让本机 Agent 直接流式读取大文件（几十 MB 的原生结果不适合整份走消息通道）。
    该字段**不得**写入任何面向浏览器或用户的响应、会话记录或日志；下载地址只用
    artifact_id。路径经登记表解析（越界即拒绝），返回前会重新核验文件摘要与登记值
    是否一致——摘要不符时拒绝定位，而不是交出一个已经变样的文件。
    """
    def call() -> dict[str, Any]:
        STORE.status(run_id)
        locator = EXPORTS.artifact_locator(run_id, artifact_id)
        return envelope(
            ok=True, status="located",
            summary=f"已定位成果 {locator['title']}（{locator['size_bytes']} 字节）",
            facts={"run_id": run_id, "artifact_id": artifact_id,
                   "size_bytes": locator["size_bytes"], "sha256": locator["sha256"],
                   "format": locator["format"]},
            warnings=["internal_path 仅供同机受信客户端内部使用，不得写入任何面向浏览器的响应"],
            next_actions=["list_hydro_artifacts"],
            data=locator,
        )
    return _guard(call)


@mcp.resource("hydro-artifact://{run_id}/{artifact_id}",
              mime_type="application/octet-stream",
              description="某次运行的一个已登记成果文件；媒体类型以成果描述为准")
def read_artifact_resource(run_id: str, artifact_id: str) -> bytes:
    """Serve one registered artifact over the resource channel.

    Only files this service registered for this run are served, and only while
    their bytes still match the recorded digest. Over the channel's size ceiling
    the read is refused, not truncated: a partially delivered scientific file is
    a wrong answer, and the same-machine adapter exists for the large ones.
    """
    raw, _media_type = EXPORTS.artifact_blob(run_id, artifact_id)
    return raw


@mcp.tool(structured_output=True)
def list_swmm_model_artifacts(run_id: str) -> dict[str, Any]:
    """列出本次运行的 SWMM 建模成果（INP / RPT / OUT / 审查表 / 雨量序列 / 子汇水区）。

    只返回公开字段（成果编号、格式、角色、大小、SHA-256、QA 状态），**不返回内部
    路径**：下载请使用返回的 artifact_id。同时说明这些成果是否齐全——必需文件
    （INP、RPT）缺失时明确列出缺什么，而不是把不完整的模型说成完整的。
    """
    def call() -> dict[str, Any]:
        STORE.status(run_id)
        verdict = EXPORTS.bundle_verdict(run_id)
        members = list(verdict["members"])
        present = [item for item in members if item.get("present")]
        return envelope(
            ok=True, status="listed",
            summary=(f"SWMM 建模成果 {len(present)} 项（完整）" if verdict["complete"]
                     else f"SWMM 建模成果 {len(present)} 项；缺少必需文件："
                          + "、".join(verdict["missing_required"])),
            facts={"run_id": run_id, "count": len(present),
                   "complete": verdict["complete"],
                   "missing_required": verdict["missing_required"]},
            warnings=([] if verdict["complete"] else
                      ["缺少必需文件：" + "、".join(verdict["missing_required"])
                       + "；不能作为完整模型成果对外提供"]),
            limitations=["这些是本次运行的原始建模文件；结果读取器目前只解析子汇水区径流变量，"
                         "未导出节点与管段时间序列"],
            next_actions=["prepare_swmm_model_bundle"],
            data={"run_id": run_id, "complete": verdict["complete"],
                  "missing_required": verdict["missing_required"],
                  "checked_at": verdict.get("checked_at"),
                  "members": [{"role": item["role"], "required": item["required"],
                               "present": item["present"],
                               "artifact": item["artifact"]} for item in members],
                  "artifacts": [item["artifact"] for item in present]},
        )
    return _guard(call)


@mcp.tool()
def get_hydro_visualization(visualization_id: str) -> Image:
    """返回已完成的预览图（PNG）。"""
    return Image(path=VISUALIZATIONS.image_path(visualization_id))


@mcp.tool(structured_output=True)
def list_visualization_artifacts(visualization_id: str) -> dict[str, Any]:
    """列出某张图对应的、本次运行真实登记的可下载成果。

    只返回公开字段：成果编号、格式、角色、大小、SHA-256、单位、CRS、QA 状态
    与来源阶段。**不返回服务器内部路径**：下载请使用返回的 artifact_id。
    关联关系是渲染时按“路径 + 哈希”记录的，不是按文件名猜的；没有关联到登记
    成果的图会说明原因，而不是列出可能是别的运行的文件。
    """
    def call() -> dict[str, Any]:
        payload = EXPORTS.artifacts_for_visualization(VISUALIZATIONS, visualization_id)
        return envelope(
            ok=payload["status"] in {"queued", "running", "succeeded"},
            status=str(payload["status"]),
            summary=f"{payload['title']}：{len(payload['artifacts'])} 项可下载成果",
            facts={"visualization_id": visualization_id, "run_id": payload["run_id"],
                   "artifacts": len(payload["artifacts"]),
                   "qa_failed": payload["qa_failed"]},
            warnings=payload["warnings"],
            limitations=payload["limitations"],
            next_actions=([] if not payload["artifacts"]
                          else ["准备导出：prepare_artifact_export"]),
            data=payload,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def compare_hydro_visualizations(
    visualization_ids: list[str], mode: str = "side_by_side"
) -> dict[str, Any]:
    """生成多图并排、栅格差值或栅格百分比变化图，并报告来源差异。"""
    def call() -> dict[str, Any]:
        result = VISUALIZATIONS.compare(visualization_ids, mode=mode)
        return envelope(
            ok=True, status="compared",
            summary=("源文件一致，可直接比较" if result["same_sources"]
                     else "存在差异：" + "；".join(result["differences"][:3])),
            facts={"run_ids": result["run_ids"], "kinds": result["kinds"],
                   "qa_failed_count": result["qa_failed_count"],
                   "comparison_visualization_id": result["comparison_visualization_id"],
                   "mode": result["mode"]},
            warnings=result["differences"],
            next_actions=["get_hydro_visualization", "get_hydro_visualization_status"],
            data=result,
        )
    return _guard(call)


@mcp.tool(structured_output=True)
def cleanup_hydro_products(identifiers: list[str], protected_ids: list[str],
                           dry_run: bool = True) -> dict[str, Any]:
    """按会话产物编号清理输入、模型、结果和图件；拒绝运行中任务，保护共享依赖。"""
    def call():
        try:
            result = clean_products(identifiers, protected_ids, dry_run=dry_run,
                                    store=STORE, visualizations=VISUALIZATIONS)
        except OSError as exc:
            raise RuntimeError("水文产物清理失败；对话已保留，请检查文件占用后重试删除。") from exc
        return envelope(ok=True, status="planned" if dry_run else "deleted",
                        summary="会话产物清理计划" if dry_run else "会话产物已清理", data=result)
    return _guard(call)


def main(argv: list[str] | None = None) -> int:
    """Start the hydro MCP.

    Recovery runs when the service starts, not when a client connects: it is the
    service that owns the workers, so it is the service that knows which of them
    are still alive.
    """
    from watershed_exchange.serve import serve

    def recover() -> None:
        STORE.restore()
        VISUALIZATIONS.restore()

    return serve(mcp, "watershed-hydro-stability-mcp", default_port=8766,
                 before_serve=recover, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
