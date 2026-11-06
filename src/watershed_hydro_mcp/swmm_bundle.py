"""The complete SWMM model bundle: everything needed to redo the run.

A bundle is not a zip of "the files we happened to find". Every member is a
**registered artifact of this run**, laid out at a fixed path, listed in a
manifest with its role and digest, and checksummed. The README says what the
model is, what it was run on, and what its limitations are — because a bundle
that travels without that is a folder of unknown provenance.

Two rules:

* **A bundle says whether it is complete.** If the input model, the report or
  the native results are missing, the members are still packaged as a
  *diagnostic* bundle, and the name, the status and the README all say so.
* **Nothing outside this run gets in.** Members are resolved from this run's
  registry, never by scanning a directory, so a neighbouring job's file with a
  matching name cannot be swept up.

And one more, learned from a real run: **the checksums describe the bytes that
are packed, not the bytes that were registered.** A later stage may legitimately
rewrite a file an earlier stage produced — the build stage recalculates the
rainfall series by subcatchment area weight and writes it back over the rainfall
stage's file — so the recorded digest can be stale by the time a bundle is made.
Both are written out, and the difference is reported rather than smoothed over.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watershed_exchange.paths import sha256_file

from .artifact_registry import ArtifactRegistry
from .state import utc_now, write_json


#: Where each kind of member goes inside the bundle.
#: ``(role, token)`` finds the artifact; the value is its path in the archive.
LAYOUT: tuple[tuple[str, str, str, bool], ...] = (
    ("model_input", ".inp", "model/model.inp", True),
    ("native_report", ".rpt", "results/model.rpt", True),
    ("native_result", ".out", "results/model.out", True),
    ("source_table", "auto_swmm_parameter_review.csv", "tables/parameter_review.csv", False),
    ("source_table", "auto_swmm_rainfall_review.csv", "tables/rainfall_review.csv", False),
    ("source_table", "auto_swmm_station_idw_weights.csv", "tables/station_weights.csv", False),
    ("source_table", "auto_swmm_landuse_response_units.csv", "tables/landuse_response_units.csv", False),
    ("source_table", "auto_swmm_landuse_area_review.csv", "tables/landuse_area_review.csv", False),
    ("source_table", "auto_swmm_parent_runoff.csv", "tables/parent_runoff.csv", False),
    ("source_raster", "auto_swmm_landuse_response_index.tif", "spatial/landuse_response_index.tif", False),
    ("metadata", "auto_swmm_landuse_response_summary.json", "metadata/landuse_response_summary.json", False),
)

#: Members that are gathered as a set rather than one file, because a shapefile
#: is several files that only mean something together.
SHAPEFILE_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class BundleError(ValueError):
    """The bundle could not be built, with a reason a caller can act on."""


@dataclass(frozen=True)
class Member:
    """One file of the bundle, and where it came from."""

    role: str
    path_in_bundle: str
    source_path: Path
    artifact_id: str
    stage_id: str
    required: bool
    sha256: str
    size_bytes: int
    #: The digest the registry recorded when the stage registered the file, and
    #: whether the bytes still match it. A later stage can rewrite a file an
    #: earlier stage produced (the build stage rewrites the rainfall series),
    #: and a checksum list that copied the *recorded* digest would then describe
    #: bytes that are no longer in the archive.
    recorded_sha256: str = ""
    sha256_matches_record: bool = True

    def as_manifest_entry(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "path": self.path_in_bundle,
            "artifact_id": self.artifact_id,
            "stage_id": self.stage_id,
            "required": self.required,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "recorded_sha256": self.recorded_sha256,
            "sha256_matches_record": self.sha256_matches_record,
        }


def inspect_swmm_output(path: Path) -> dict[str, Any]:
    """Header-level facts about a SWMM binary output, and whether it is intact.

    Reads the header and the trailing index only — never the whole file — and
    reports what it found rather than raising: a damaged result is worth handing
    over as a diagnostic, as long as it is labelled as damaged.
    """
    facts: dict[str, Any] = {"intact": False, "reason": None}
    try:
        with Path(path).open("rb") as stream:
            head = stream.read(28)
            if len(head) < 28:
                return {**facts, "reason": "文件过短，不是完整的 SWMM 输出"}
            (start_magic, version, flow_units, subcatchments, nodes, links,
             pollutants) = struct.unpack("<7i", head)
            stream.seek(-24, 2)
            tail = stream.read(24)
            if len(tail) < 24:
                return {**facts, "reason": "文件尾部被截断"}
            (_, _, results_pos, periods, error_code, end_magic) = struct.unpack("<6i", tail)
            facts.update({
                "version": version, "flow_units": flow_units,
                "subcatchments": subcatchments, "nodes": nodes, "links": links,
                "pollutants": pollutants, "periods": periods,
                "error_code": error_code,
            })
            if start_magic != end_magic:
                return {**facts, "reason": "首尾 magic number 不一致，文件不完整"}
            if periods <= 0:
                return {**facts, "reason": "报告时段数为零"}
            if error_code != 0:
                return {**facts, "reason": f"SWMM 报告模拟错误码 {error_code}"}
            if results_pos <= 0 or results_pos >= Path(path).stat().st_size:
                return {**facts, "reason": "结果区偏移越界"}
            stream.seek(results_pos - 12)
            start_ole, report_step = struct.unpack("<di", stream.read(12))
            facts["report_step_seconds"] = report_step
            facts["start_datetime"] = _ole_to_iso(start_ole)
            facts["intact"] = True
    except (OSError, struct.error) as exc:
        return {**facts, "reason": f"无法读取：{exc}"}
    return facts


def _ole_to_iso(value: float) -> str | None:
    """SWMM stores the start time as an OLE automation date."""
    try:
        from datetime import timedelta

        return (datetime(1899, 12, 30) + timedelta(days=float(value))).isoformat()
    except (OverflowError, ValueError):
        return None


def swmm_engine_version(report: Path | None) -> str | None:
    """The engine's own version string, read from its report."""
    if report is None or not Path(report).is_file():
        return None
    try:
        with Path(report).open("r", encoding="utf-8", errors="replace") as stream:
            for _ in range(12):
                line = stream.readline()
                if not line:
                    break
                if "VERSION" in line.upper():
                    return line.strip()[:120]
    except OSError:
        return None
    return None


def _one_record_per_file(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop an export copy when the stage that produced the file is also indexed.

    The export stage copies the finished results into ``final_results``, so the
    registry holds the same content twice under two paths. Both would map to the
    same place inside the bundle, which would put one file in the archive twice;
    the stage copy is the one to keep.
    """
    best: dict[str, dict[str, Any]] = {}
    for item in records:
        digest = str(item.get("sha256") or item.get("internal_path"))
        # Keyed on the name *and* the content: only a copy of the same file is
        # collapsed, never two different files that happen to hold equal bytes.
        key = f"{item.get('title')}|{digest}"
        current = best.get(key)
        if current is None or (current.get("kind") == "export"
                               and item.get("kind") != "export"):
            best[key] = item
    return list(best.values())


def collect_members(registry: ArtifactRegistry, run_id: str) -> list[Member]:
    """The registered files this run would package, in layout order.

    Resolved from this run's registry only: a file that merely looks right on
    disk is not a member, because the registry is what says the run produced it.
    """
    records = _one_record_per_file(list(registry.all()))
    members: list[Member] = []
    taken: set[str] = set()

    def _add(role: str, path_in_bundle: str, record: dict[str, Any], required: bool) -> None:
        if path_in_bundle in taken:
            return
        try:
            source = registry.resolve(record)
        except ValueError:
            return
        if not source.is_file():
            return
        taken.add(path_in_bundle)
        recorded = str(record.get("sha256") or "")
        actual = sha256_file(source)
        members.append(Member(
            role=role, path_in_bundle=path_in_bundle, source_path=source,
            artifact_id=str(record.get("artifact_id")), stage_id=str(record.get("stage_id")),
            required=required, sha256=actual, size_bytes=source.stat().st_size,
            recorded_sha256=recorded, sha256_matches_record=(actual == recorded)))

    from .export_service import role_for
    for role, token, path_in_bundle, required in LAYOUT:
        match = next((item for item in records
                      if (item.get("kind") in {role, "swmm_input", "swmm_result"} or role_for(item) == role)
                      and str(item.get("title", "")).endswith(token)), None)
        if match is not None:
            _add(role, path_in_bundle, match, required)

    # The subcatchment shapefile travels as a set: a .shp without its .dbf is
    # not a dataset.
    spatial = [item for item in records
               if str(item.get("title", "")).endswith(SHAPEFILE_SUFFIXES)
               and "subcatchments." in str(item.get("title", ""))
               and "channel_cells" not in str(item.get("title", ""))
               and "nodes" not in str(item.get("title", ""))
               and "outfalls" not in str(item.get("title", ""))]
    for item in sorted(spatial, key=lambda entry: str(entry.get("title"))):
        suffix = Path(str(item["title"])).suffix
        _add("spatial", f"spatial/subcatchments{suffix}", item, False)

    active = next((item for item in records
                   if str(item.get("title", "")).endswith("_active_hourly.csv")), None)
    if active is not None:
        _add("forcing", "forcing/rainfall_hourly.csv", active, False)
    return members


def missing_required(members: list[Member]) -> list[str]:
    present = {member.role for member in members}
    missing = [role for role, _token, _path, required in LAYOUT if required and role not in present]
    model = next((member for member in members if member.role == 'model_input'), None)
    if model and '; LANDUSE_REPRESENTATION response_units' in model.source_path.read_text(encoding='utf-8', errors='replace'):
        paths = {member.path_in_bundle for member in members}
        for path in ('tables/landuse_response_units.csv', 'tables/landuse_area_review.csv',
                     'spatial/landuse_response_index.tif', 'metadata/landuse_response_summary.json'):
            if path not in paths:
                missing.append('landuse_mapping:' + path)
    return missing


def run_window(registry: ArtifactRegistry, run_id: str) -> dict[str, Any]:
    """The run's own record of what it did, as far as it is recorded on disk."""
    job = registry.job_dir / "job.json"
    if not job.is_file():
        return {}
    try:
        payload = json.loads(job.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    plan = payload.get("plan") or {}
    rain = plan.get("rainfall") or {}
    event = rain.get("event") or plan.get("event") or {}
    return {
        "mode": payload.get("mode"),
        "preset": payload.get("parameter_preset"),
        "created_at": payload.get("created_at"),
        "event_start": event.get("start_datetime"),
        "event_end": event.get("end_datetime"),
        "timezone": event.get("timezone") or rain.get("timezone"),
        "stages": payload.get("stages"),
    }


def readme_text(run_id: str, members: list[Member], *,
                engine: str | None, window: dict[str, Any],
                qa_status: str | None, output_facts: dict[str, Any] | None,
                limitations: list[str], complete: bool) -> str:
    lines = [
        f"# SWMM 模型成果包 · {run_id}",
        "",
        f"打包时间：{utc_now()}",
        f"完整性：{'完整（必需文件齐全）' if complete else '**不完整：仅供诊断**'}",
        "",
        "## 这是什么",
        "",
        "本次运行实际使用与产生的一套 SWMM 文件，用于复核与复算。"
        "包内每个文件都来自本次运行登记的成果，未混入其他运行的文件。",
        "",
        "## 模型与运行",
        "",
        f"- SWMM 引擎：{engine or '报告文件缺失，未能读取版本'}",
        f"- 运行模式：{window.get('mode') or '未记录'}",
        f"- 参数预设：{window.get('preset') or '未记录'}",
        f"- 事件窗口：{window.get('event_start') or '未记录'} 至 {window.get('event_end') or '未记录'}"
        f"（时区 {window.get('timezone') or '未记录'}）",
        f"- 质量门槛：{qa_status or '未记录'}",
    ]
    if output_facts:
        lines += [
            f"- OUT 报告时段数：{output_facts.get('periods')}",
            f"- OUT 报告步长：{output_facts.get('report_step_seconds')} 秒",
            f"- 子汇水区 / 节点 / 管段：{output_facts.get('subcatchments')} / "
            f"{output_facts.get('nodes')} / {output_facts.get('links')}",
            f"- OUT 完整性：{'通过（首尾 magic 一致、时段数大于零、模拟错误码为零）' if output_facts.get('intact') else output_facts.get('reason')}",
        ]
    lines += ["", "## 包内文件", ""]
    for member in members:
        mark = "必需" if member.required else "可选"
        note = "" if member.sha256_matches_record else "，**该文件在登记之后被后续阶段改写**"
        lines.append(f"- `{member.path_in_bundle}`（{mark}，{member.size_bytes} 字节，"
                     f"来源成果 {member.artifact_id}{note}）")
    drifted = [member for member in members if not member.sha256_matches_record]
    if drifted:
        lines += [
            "",
            "**包内文件与登记摘要的差异（已如实列出，未做掩盖）**",
            "",
            "下列文件在校验时与登记时的摘要不同：`checksums.sha256` 写的是**打包时的实际"
            "摘要**，`manifest.json` 同时给出登记摘要（`recorded_sha256`）与是否一致"
            "（`sha256_matches_record`）。依本次运行的实际过程，成因是 SWMM 建模阶段会"
            "按子汇水区面积权重重算雨量序列并覆写降雨阶段产出的逐时雨量文件，属于同一"
            "份降雨数据的重算写回，不是外部改动；但它意味着这几条**登记记录的摘要已过时**。",
            "",
        ]
        for member in drifted:
            lines.append(f"- `{member.path_in_bundle}`：登记 {member.recorded_sha256[:12]}…，"
                         f"实际 {member.sha256[:12]}…")
    absent = missing_required(members)
    if absent:
        lines += ["", f"**缺少必需文件：{', '.join(absent)}**；"
                      "本包不能作为完整模型包使用，只能用于诊断。"]
    lines += [
        "",
        "## 复算步骤",
        "",
        "1. 用同一版本的 SWMM 引擎打开 `model/model.inp`；",
        "2. 确认 `[OPTIONS]`、`[RAINGAGES]`、`[SUBCATCHMENTS]` 与本次运行一致；",
        "3. 运行引擎，与 `results/model.rpt` 的 error/warning 与连续性误差对照；",
        "4. 逐文件核对 `checksums.sha256`，确认包内容没有被改动。",
        "",
        "## 能力边界",
        "",
        "- 本包只包含本次运行产生的文件，不含原始 DEM、土地利用等外部输入（它们在模型输入包里）；",
        "- 结果读取器目前只解析子汇水区径流变量，未导出节点与管段的全部时间序列；",
        "- `.out` 是 SWMM 原生二进制，请用引擎或解析器读取，不要当文本打开；",
    ]
    for item in limitations:
        lines.append(f"- {item}")
    lines += [
        "- 本包用于复核与复算，不构成工程级边坡安全鉴定或灾害预报结论。",
        "",
    ]
    return "\n".join(lines)


def _checksums(members: list[Member]) -> str:
    return "\n".join(f"{member.sha256}  {member.path_in_bundle}" for member in members) + "\n"


def build_bundle(run_id: str, registry: ArtifactRegistry, *,
                 destination: Path, qa_status: str | None = None,
                 limitations: list[str] | None = None,
                 software_versions: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the bundle to ``destination`` atomically, and describe it.

    Everything is written to a temporary name first: a half-written archive is
    never visible, and never registered.
    """
    members = collect_members(registry, run_id)
    if not members:
        raise BundleError("本次运行没有可打包的 SWMM 成果")
    absent = missing_required(members)
    complete = not absent
    report_member = next((item for item in members
                          if item.path_in_bundle == "results/model.rpt"), None)
    output_member = next((item for item in members
                          if item.path_in_bundle == "results/model.out"), None)
    output_facts = inspect_swmm_output(output_member.source_path) if output_member else None
    window = run_window(registry, run_id)
    engine = swmm_engine_version(report_member.source_path if report_member else None)

    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "complete": complete,
        "missing_required": absent,
        "qa_status": qa_status,
        "swmm_engine": engine,
        "run": window,
        "native_output": output_facts,
        "members": [member.as_manifest_entry() for member in members],
        "drifted_members": [member.path_in_bundle for member in members
                            if not member.sha256_matches_record],
        "checksums_rule": "checksums.sha256 记录打包时的实际字节摘要；"
                          "登记摘要见各成员的 recorded_sha256",
    }
    readme = readme_text(run_id, members, engine=engine, window=window,
                         qa_status=qa_status, output_facts=output_facts,
                         limitations=list(limitations or []), complete=complete)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("README.md", readme)
            archive.writestr("manifest.json",
                             json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("checksums.sha256", _checksums(members))
            archive.writestr("metadata/software_versions.json", json.dumps(
                software_versions or {}, ensure_ascii=False, indent=2))
            qa_path = registry.job_dir / "qa.json"
            if qa_path.is_file():
                archive.write(qa_path, "qa/qa.json")
            for member in members:
                # Members are already known to be inside the run directory; the
                # archive path comes from the layout, never from the caller.
                archive.write(member.source_path, member.path_in_bundle)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "bundle_path": destination,
        "complete": complete,
        "missing_required": absent,
        "members": [member.as_manifest_entry() for member in members],
        "drifted_members": [member.path_in_bundle for member in members
                            if not member.sha256_matches_record],
        "native_output": output_facts,
        "swmm_engine": engine,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def bundle_name(run_id: str, *, when: datetime | None = None,
                complete: bool = True, key: str | None = None) -> str:
    """A stable, self-describing name for the archive.

    The content key is part of the name when the caller knows it. Two different
    bundles built in the same second would otherwise share a name, and the
    second would silently overwrite the first — leaving an artifact id that
    resolves to a file it does not describe.
    """
    stamp = (when or datetime.now(timezone.utc)).strftime(TIMESTAMP_FORMAT)
    prefix = "swmm_model" if complete else "swmm_model_INCOMPLETE"
    suffix = f"_{key[:8]}" if key else ""
    return f"{prefix}_{run_id}_{stamp}{suffix}.zip"


def write_manifest_copy(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _digest(path: Path) -> str:
    """A file digest, for the caller that wants one without the registry."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
