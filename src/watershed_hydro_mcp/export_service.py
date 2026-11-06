"""What a caller may download, and how it is described.

Everything a browser receives about an artifact comes from here, and it is built
by projection rather than by filtering: the registry record is *reduced* to the
fields below, so a field the registry adds later is not published by default.
``internal_path`` is never among them — a download is addressed by opaque id, and
a caller that cannot see a path cannot ask for one.

The rules that shape a descriptor:

* the **format** comes from the suffix, because that is what the bytes are;
* the **role** says what the file is *to the person downloading it* (a source
  raster, a model input, a native result), which is not the same thing as the
  stage that happened to write it;
* what can be downloaded **now** is separated from what can be **generated on
  request**, so a button is never offered for a file that does not exist yet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from watershed_exchange.paths import PathSafetyError

from .artifact_registry import ArtifactRegistry
from .settings import SETTINGS, Settings
from .state import read_json, utc_now


#: The format name a caller asks for and a button is labelled with.
FORMAT_BY_SUFFIX: dict[str, str] = {
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".geojson": "geojson",
    ".md": "markdown",
    ".txt": "text",
    ".log": "text",
    ".prj": "text",
    ".cpg": "text",
    ".inp": "inp",
    ".rpt": "rpt",
    ".out": "out",
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".pdf": "pdf",
    ".zip": "zip",
    ".xlsx": "xlsx",
    ".shp": "shp",
    ".shx": "shapefile-part",
    ".dbf": "shapefile-part",
}

#: What the file is to whoever downloads it.
ROLE_BY_SUFFIX: dict[str, str] = {
    ".tif": "source_raster",
    ".tiff": "source_raster",
    ".csv": "source_table",
    ".tsv": "source_table",
    ".inp": "model_input",
    ".rpt": "native_report",
    ".out": "native_result",
    ".zip": "package",
    ".png": "figure",
    ".jpg": "figure",
    ".jpeg": "figure",
    ".pdf": "figure",
    ".shp": "vector",
    ".shx": "vector",
    ".dbf": "vector",
    ".json": "metadata",
    ".geojson": "vector",
    ".md": "documentation",
    ".txt": "documentation",
    ".log": "documentation",
    ".prj": "vector",
    ".cpg": "vector",
    ".xlsx": "spreadsheet",
}

#: Formats that can be produced from what is already on disk. Only conversions
#: that lose nothing are listed: a CSV becomes a workbook of the same numbers, a
#: preview PNG never becomes a raster.
DERIVABLE_FROM: dict[str, tuple[str, ...]] = {
    "csv": ("xlsx",),
    "tsv": ("xlsx",),
}

#: Formats a caller may ask for by name.
EXPORTABLE_FORMATS: frozenset[str] = frozenset({"xlsx"})

#: The version of the bundle layout and its manifest. A new layout must not be
#: mistaken for an old one, so it is recorded on every bundle and checked before
#: an existing archive is reused.
BUNDLE_VERSION = "1.0"

#: The most one resource read may carry. The MCP resource channel answers in a
#: single JSON-RPC response, so the whole file is built in memory on the server
#: and encoded on the way out; this is a hard ceiling, not a preference. Larger
#: files are streamed from disk by the same-machine adapter instead.
MAX_RESOURCE_BYTES = 32 * 1024 * 1024

#: Fields of a registry record that may leave the service.
PUBLIC_FIELDS: tuple[str, ...] = (
    "artifact_id", "title", "media_type", "size_bytes", "sha256", "stage_id",
    "kind", "units", "crs", "bbox", "qa_status", "created_at",
    "width", "height", "count", "dtype", "nodata", "transform", "resolution",
    "columns", "column_count", "rows", "rows_at_least", "time_column",
    "derived_from_artifact_ids", "producer", "producer_version",
)


def _fresh_media_type(record: dict[str, Any]) -> str:
    """The media type for a record, derived now rather than remembered.

    A record is written once, and this machine used to derive ``.csv`` from the
    Windows registry, so an older run can still say ``application/vnd.ms-excel``
    for a comma-separated file. Both channels therefore state the type they are
    sending *now*; the record keeps its history.
    """
    from .artifact_registry import media_type as media_type_for

    title = str(record.get("title") or "")
    if title:
        return media_type_for(Path(title))
    return str(record.get("media_type") or "application/octet-stream")


def format_for(record: dict[str, Any]) -> str:
    suffix = Path(str(record.get("title") or "")).suffix.lower()
    return FORMAT_BY_SUFFIX.get(suffix, "binary")


def role_for(record: dict[str, Any]) -> str:
    suffix = Path(str(record.get("title") or "")).suffix.lower()
    return ROLE_BY_SUFFIX.get(suffix, "file")


def public_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    """The view of one artifact that a caller may see.

    Built by projection: only the declared fields travel, and ``internal_path``
    is not one of them. What can be downloaded now and what can be generated on
    request are stated separately so the interface never offers a file that is
    not there.

    The media type is recomputed from the title rather than copied from the
    record. A record is written once, and this machine used to derive ``.csv``
    from the Windows registry — a run registered then would still hand a browser
    ``application/vnd.ms-excel`` for a comma-separated file, and the browser
    would save it as ``.xls``. The bytes have not changed; what changes is how
    they should be treated.
    """
    from .artifact_registry import media_type as media_type_for

    descriptor = {field: record.get(field) for field in PUBLIC_FIELDS if field in record}
    title = str(record.get("title") or "")
    if title:
        descriptor["media_type"] = media_type_for(Path(title))
        # Registries written before the export UI existed do not carry units.
        # The filename tokens are the same deterministic convention used when
        # registering new artifacts, so old and new runs describe the same
        # scientific quantity consistently without rewriting their history.
        if not descriptor.get("units"):
            from .artifact_registry import units_for

            descriptor["units"] = units_for(Path(title))
    descriptor["format"] = format_for(record)
    descriptor["role"] = role_for(record)
    descriptor["ready"] = bool(record.get("ready"))
    descriptor["formats"] = ["original"] if record.get("ready") else []
    descriptor["exportable"] = list(DERIVABLE_FROM.get(descriptor["format"], ()))
    return descriptor


class ExportError(ValueError):
    """An export could not be produced, with a reason a caller can act on."""


class ExportService:
    """Turning what a run registered into what a caller may download."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings

    # ------------------------------------------------------------- registry
    def registry(self, run_id: str) -> ArtifactRegistry:
        job_dir = self.settings.job_dir(run_id)
        return ArtifactRegistry(job_dir, run_id, job_dir / "project")

    def run_qa_status(self, run_id: str) -> str | None:
        """The run's quality-gate verdict, from the report the gates wrote."""
        path = self.settings.job_dir(run_id) / "qa.json"
        if not path.is_file():
            return None
        try:
            report = read_json(path)
        except ValueError:
            return None
        return "passed" if report.get("passed") else "failed"

    def descriptor(self, run_id: str, record: dict[str, Any]) -> dict[str, Any]:
        descriptor = public_descriptor(record)
        if descriptor.get("qa_status") is None:
            # Older registries were written before the gates had run; the
            # verdict is a property of the run and is read from its report.
            descriptor["qa_status"] = self.run_qa_status(run_id)
        return descriptor

    @staticmethod
    def _find(registry: ArtifactRegistry, artifact_id: str) -> dict[str, Any] | None:
        """A registered artifact, or ``None`` — the registry's own lookup raises."""
        try:
            return registry.by_id(artifact_id)
        except ValueError:
            return None

    def descriptors(self, run_id: str, artifact_ids: list[str]) -> list[dict[str, Any]]:
        registry = self.registry(run_id)
        found: list[dict[str, Any]] = []
        for identifier in artifact_ids:
            record = self._find(registry, identifier)
            if record is None:
                continue
            found.append(self.descriptor(run_id, record))
        return found

    # ------------------------------------------------------ reading one file
    def _ready_file(self, run_id: str, artifact_id: str) -> tuple[dict[str, Any], Path]:
        """The record and the file it names, or a reason a caller can act on.

        Every failure is distinct on purpose: "no such artifact", "not finished
        yet", "the registry's path is outside the run" and "the bytes are not
        the bytes the record describes" mean different things to whoever is
        asking, and collapsing them would hide a real integrity problem.
        """
        registry = self.registry(run_id)
        record = self._find(registry, artifact_id)
        if record is None:
            raise ExportError(f"成果不存在：{artifact_id}")
        if not record.get("ready"):
            raise ExportError("成果尚未写入完成，暂不能下载")
        try:
            path = registry.resolve(record)
        except (PathSafetyError, ValueError) as exc:
            raise ExportError(f"成果路径不合法：{exc}") from exc
        if not path.is_file():
            raise ExportError("成果文件已不存在")
        return record, path

    def artifact_size(self, run_id: str, artifact_id: str) -> int:
        """How many bytes the registered file holds right now."""
        _record, path = self._ready_file(run_id, artifact_id)
        return path.stat().st_size

    def artifact_blob(self, run_id: str, artifact_id: str,
                      *, limit_bytes: int = MAX_RESOURCE_BYTES) -> tuple[bytes, str]:
        """One registered file's bytes, for the resource channel.

        The digest is recomputed before answering: a resource that hands over
        bytes the registry does not describe would let a stale or rewritten file
        travel under a registered name. Over the limit is refused rather than
        truncated, because a truncated scientific file is a wrong answer.
        """
        from watershed_exchange.paths import sha256_file

        record, path = self._ready_file(run_id, artifact_id)
        size = path.stat().st_size
        if size > limit_bytes:
            raise ExportError(
                f"文件 {size} 字节，超过资源通道上限 {limit_bytes} 字节；"
                "请使用同机下载通道（locate_artifact）流式获取")
        raw = path.read_bytes()
        recorded = str(record.get("sha256") or "")
        # The digest is computed from the bytes actually read, so an empty
        # record still yields a verified answer and a rewritten file cannot
        # travel under a registered name.
        if recorded and hashlib.sha256(raw).hexdigest() != recorded:
            raise ExportError("成果文件与登记摘要不一致，已拒绝下载；请重新登记该成果后重试")
        return raw, _fresh_media_type(record)

    def artifact_locator(self, run_id: str, artifact_id: str) -> dict[str, Any]:
        """Where one registered file lives on this machine.

        For the same-machine adapter only: it is what lets a large file be
        streamed from disk instead of travelling through the message channel in
        one piece. It carries an absolute path, so it is returned to a trusted
        local client and must never reach a browser response, a session file or
        a log — the caller is responsible for that, and this docstring is the
        contract.

        The path is the registry's own resolution (which refuses anything
        outside the run), the file must be ``ready``, and the digest is verified
        before the locator is handed over: a locator that does not name the
        bytes the record describes is worse than no locator at all.
        """
        from watershed_exchange.paths import sha256_file

        record, path = self._ready_file(run_id, artifact_id)
        recorded = str(record.get("sha256") or "")
        digest = sha256_file(path)
        if recorded and digest != recorded:
            raise ExportError("成果文件与登记摘要不一致，已拒绝下载；请重新登记该成果")
        return {
            "artifact_id": str(record.get("artifact_id")),
            "run_id": run_id,
            "title": record.get("title"),
            "filename": Path(str(record.get("title") or "artifact.bin")).name,
            "internal_path": str(path),
            "sha256": digest,
            "recorded_sha256": recorded,
            "size_bytes": path.stat().st_size,
            "media_type": _fresh_media_type(record),
            "format": format_for(record),
            "role": role_for(record),
            "qa_status": record.get("qa_status") or self.run_qa_status(run_id),
            "ready": True,
        }

    # --------------------------------------------------------- presentation
    def artifacts_for_visualization(self, service, visualization_id: str) -> dict[str, Any]:
        """The figure, and the real files it was drawn from.

        Every file here is a registered artifact of this session's own run: the
        association was recorded when the figure was rendered (path *and*
        digest), or resolved from an older record on first read. Nothing is
        inferred from a file name.
        """
        record = service.status(visualization_id)
        run_id = str(record.get("run_id") or "")
        artifacts = self.descriptors(run_id, list(record.get("source_artifact_ids") or []))
        return {
            "visualization_id": record.get("visualization_id") or visualization_id,
            "run_id": run_id,
            "kind": record.get("kind"),
            "title": record.get("title"),
            "status": record.get("status"),
            "qa_failed": bool(record.get("qa_failed")),
            "limitations": list(record.get("limitations") or []),
            "image_sha256": record.get("image_sha256"),
            "sources": [
                {"scope": item.get("scope"), "sha256": item.get("sha256"),
                 "size_bytes": item.get("size_bytes")}
                for item in (record.get("sources") or []) if item.get("scope")
            ],
            "artifacts": artifacts,
            "warnings": list(record.get("warnings") or [])
            + ([] if artifacts else ["该图没有可下载的登记成果"]),
        }

    def swmm_bundle_members(self, run_id: str) -> list[dict[str, Any]]:
        """The files a complete SWMM model bundle would contain, and whether
        each is present. Missing required members are reported, never hidden."""
        registry = self.registry(run_id)
        wanted = (
            ("model_input", ".inp", True),
            ("native_report", ".rpt", True),
            ("native_result", ".out", True),
            ("source_table", "auto_swmm_parameter_review.csv", False),
            ("source_table", "auto_swmm_rainfall_review.csv", False),
            ("source_table", "auto_swmm_landuse_response_units.csv", False),
            ("source_table", "auto_swmm_landuse_area_review.csv", False),
            ("source_raster", "auto_swmm_landuse_response_index.tif", False),
            ("metadata", "auto_swmm_landuse_response_summary.json", False),
        )
        members: list[dict[str, Any]] = []
        records = list(registry.all())
        for role, token, required in wanted:
            match = next((item for item in records
                          if role_for(item) == role and str(item.get("title", "")).endswith(token)),
                         None)
            members.append({
                "role": role,
                "token": token,
                "required": required,
                "present": match is not None,
                "artifact": self.descriptor(run_id, match) if match else None,
            })
        return members

    def bundle_verdict(self, run_id: str) -> dict[str, Any]:
        members = self.swmm_bundle_members(run_id)
        missing_required = [item["token"] for item in members
                            if item["required"] and not item["present"]]
        return {
            "complete": not missing_required,
            "missing_required": missing_required,
            "members": members,
            "checked_at": utc_now(),
        }

    # ------------------------------------------------------------- exports
    def _derived_path(self, run_id: str, artifact_id: str, export_format: str) -> Path:
        project = self.settings.job_dir(run_id) / "project"
        return project / "exports" / f"{artifact_id}-{export_format}.{export_format}"

    def _reusable(self, registry: ArtifactRegistry, record: dict[str, Any],
                  export_format: str) -> dict[str, Any] | None:
        """An existing derived file that is still the right one.

        Reuse requires the same source artifact, the same converter version and
        a file whose digest still matches its record: a workbook made from a
        file that has since changed is not the same workbook, so it is not
        reused under the old name.
        """
        from .workbook import CONVERTER_VERSION

        for item in registry.all():
            if item.get("derived_from_artifact_ids") != [record["artifact_id"]]:
                continue
            if not str(item.get("title") or "").endswith("." + export_format):
                continue
            if item.get("producer_version") != CONVERTER_VERSION:
                continue
            candidate = (self.settings.job_dir(str(record["hydro_run_id"]))
                         / "project" / str(item.get("internal_path") or ""))
            if not candidate.is_file():
                continue
            from watershed_exchange.paths import sha256_file

            if sha256_file(candidate) == item.get("sha256"):
                return item
        return None

    def prepare_export(self, run_id: str, artifact_id: str,
                       export_format: str) -> dict[str, Any]:
        """Generate — or reuse — a derived file for one registered artifact.

        Only conversions that lose nothing are allowed, and the result is
        registered like any other artifact: it carries the source digest, the
        converter and its version, so a reader can tell a convenience copy from
        a computed result.
        """
        from watershed_exchange.paths import sha256_file

        from .workbook import CONVERTER_VERSION, WorkbookTooLarge, build_workbook

        wanted = str(export_format or "").strip().lower()
        if wanted not in EXPORTABLE_FORMATS:
            raise ExportError(f"不支持的导出格式：{export_format}；"
                              f"允许值：{sorted(EXPORTABLE_FORMATS)}")
        registry = self.registry(run_id)
        record = self._find(registry, artifact_id)
        if record is None:
            raise ExportError(f"成果不存在：{artifact_id}")
        source_format = format_for(record)
        if wanted not in DERIVABLE_FROM.get(source_format, ()):
            raise ExportError(f"{source_format} 不能导出为 {wanted}")

        reusable = self._reusable(registry, record, wanted)
        if reusable is not None:
            return {"reused": True, "artifact": self.descriptor(run_id, reusable),
                    "source_artifact_id": artifact_id}

        try:
            source = registry.resolve(record)
        except ValueError as exc:
            raise ExportError(f"成果路径超出运行目录：{exc}") from exc
        if not source.is_file():
            raise ExportError("源文件已不存在，无法生成导出件")
        target = self._derived_path(run_id, artifact_id, wanted)
        try:
            table = build_workbook(source, target, metadata={
                "run_id": run_id,
                "artifact_id": artifact_id,
                "source_title": record.get("title"),
                "source_sha256": record.get("sha256"),
                "qa_status": record.get("qa_status") or self.run_qa_status(run_id),
                "units": record.get("units"),
                "crs": record.get("crs"),
            })
        except WorkbookTooLarge as exc:
            raise ExportError(str(exc)) from exc
        derived = registry.register(
            target, "export", kind="export", derived_from=[artifact_id],
            producer="watershed_hydro_mcp.workbook", producer_version=CONVERTER_VERSION,
            derivation={"converter": "csv_to_xlsx", "from": source_format,
                        "source_sha256": record.get("sha256")})
        if derived is None:  # pragma: no cover - the file was just written
            raise ExportError("导出件写入失败")
        assert sha256_file(target) == derived["sha256"]
        return {"reused": False, "artifact": self.descriptor(run_id, derived),
                "source_artifact_id": artifact_id, "table": table}

    # ---------------------------------------------------------- model bundle
    @staticmethod
    def _bundle_warnings(members: list[Any], missing: list[str]) -> list[str]:
        """What a caller must be told about this bundle before trusting it.

        Two things are said plainly rather than left for the reader to discover:
        a required member that is absent (the archive is not a complete model
        bundle), and a member whose bytes no longer match the digest recorded
        when its stage registered it.
        """
        warnings: list[str] = []
        if missing:
            warnings.append("缺少必需文件：" + "、".join(missing)
                            + "；本包只能作为诊断件，不能作为完整模型包使用")
        drifted = [member.path_in_bundle for member in members
                   if not member.sha256_matches_record]
        if drifted:
            warnings.append("以下文件在登记之后被后续阶段改写，其登记摘要已过时（包内 "
                            "checksums.sha256 写的是打包时的实际摘要）：" + "、".join(drifted))
        return warnings

    def prepare_swmm_bundle(self, run_id: str) -> dict[str, Any]:
        """Build — or reuse — the complete SWMM model bundle for one run.

        Reuse is keyed on the digests of every member: if any source file has
        changed since the last bundle, that bundle no longer describes this run
        and a new one is written under a new name and digest.
        """
        from . import swmm_bundle as bundles

        registry = self.registry(run_id)
        members = bundles.collect_members(registry, run_id)
        if not members:
            raise ExportError("本次运行没有可打包的 SWMM 成果")
        absent = bundles.missing_required(members)
        complete = not absent
        warnings = self._bundle_warnings(members, absent)

        key = swmm_bundle_key(members)
        cached = self._find_bundle(registry, key, complete)
        if cached is not None:
            return {"reused": True, "complete": complete, "artifact": self.descriptor(run_id, cached),
                    "missing_required": absent, "warnings": warnings,
                    "members": [member.as_manifest_entry() for member in members]}

        name = bundles.bundle_name(run_id, complete=complete, key=key)
        target = self.settings.job_dir(run_id) / "project" / "exports" / name
        try:
            result = bundles.build_bundle(
                run_id, registry, destination=target,
                qa_status=self.run_qa_status(run_id),
                limitations=["结果读取器目前只解析子汇水区径流变量，未导出节点与管段时间序列"],
                software_versions={
                    "agent_hydro_service": _module_version("watershed_hydro_mcp"),
                    "builder": bundles.__name__,
                    "built_at": utc_now(),
                })
        except bundles.BundleError as exc:
            raise ExportError(str(exc)) from exc

        record = registry.register(
            result["bundle_path"], "export", kind="export", title=name,
            derived_from=[member.artifact_id for member in members],
            producer="watershed_hydro_mcp.swmm_bundle",
            producer_version=BUNDLE_VERSION,
            derivation={"bundle_key": key, "complete": complete,
                        "members": len(members), "missing_required": absent,
                        "drifted_members": result["drifted_members"]})
        if record is None:  # pragma: no cover - the archive was just written
            raise ExportError("模型包登记失败")
        return {"reused": False, "complete": complete, "artifact": self.descriptor(run_id, record),
                "missing_required": absent, "members": result["members"],
                "drifted_members": result["drifted_members"], "warnings": warnings,
                "native_output": result["native_output"], "swmm_engine": result["swmm_engine"]}

    def _find_bundle(self, registry: ArtifactRegistry, key: str,
                     complete: bool) -> dict[str, Any] | None:
        for item in registry.all():
            derivation = item.get("derivation") or {}
            if derivation.get("bundle_key") != key:
                continue
            if bool(derivation.get("complete")) != complete:
                continue
            if item.get("producer_version") != BUNDLE_VERSION:
                continue
            try:
                path = registry.resolve(item)
            except ValueError:
                continue
            if path.is_file():
                from watershed_exchange.paths import sha256_file

                if sha256_file(path) == item.get("sha256"):
                    return item
        return None


def swmm_bundle_key(members: list[Any]) -> str:
    """A digest of the bundle's contents, used to decide whether to reuse one."""
    import hashlib

    payload = "\n".join(sorted(f"{member.path_in_bundle}:{member.sha256}"
                               for member in members))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_version(name: str) -> str | None:
    try:
        import importlib

        return str(getattr(importlib.import_module(name), "__version__", "") or "") or None
    except ImportError:  # pragma: no cover - the package is always importable here
        return None
