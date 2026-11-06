"""Background rendering of catalog visualizations.

Rendering runs off the request path with its own concurrency budget, so a large
figure never blocks a status check. Each result records the source files and
their SHA-256 digests, and a figure drawn from a `qa_failed` run is stamped so
it cannot be read as a validated result.
"""

from __future__ import annotations

import threading
import traceback
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import rasterio

from watershed_exchange.paths import sha256_file, safe_id

from ..artifact_registry import ArtifactRegistry, load_artifacts
from ..settings import SETTINGS, Settings
from ..state import new_id, read_json, replace_with_retry, utc_now, write_json
from . import catalog
from .renderer import RENDERER_VERSION, RenderContext, RENDERERS, _save, _stamp, _style


logger = logging.getLogger("watershed_hydro_mcp.visualization")


VISUALIZATION_ID_PREFIX = "viz-"

MAXIMUM_CONCURRENT_VISUALIZATIONS = 2

TERMINAL = {"succeeded", "failed"}


class VisualizationError(ValueError):
    """Raised when a visualization cannot be created or retrieved."""


class VisualizationService:
    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._active: set[str] = set()

    # ------------------------------------------------------------------ paths
    def directory(self, visualization_id: str) -> Path:
        safe_id(visualization_id, "可视化编号")
        return self.settings.visualizations_root / visualization_id

    def _record_path(self, visualization_id: str) -> Path:
        return self.directory(visualization_id) / "visualization.json"

    # ---------------------------------------------------------------- reading
    def status(self, visualization_id: str) -> dict[str, Any]:
        path = self._record_path(visualization_id)
        if not path.is_file():
            raise VisualizationError(f"可视化任务不存在：{visualization_id}")
        record = read_json(path)
        # Records written before figures carried artifact links are filled in on
        # first read, so an older run can still offer its source files without
        # being regenerated. Written back once; a second read is a plain read.
        if "source_artifact_ids" not in record and record.get("sources"):
            record["source_artifact_ids"] = self._artifact_links(
                str(record.get("run_id") or ""), record.get("sources") or [])
            try:
                write_json(path, record)
            except OSError:  # pragma: no cover - a read must not fail on a write
                pass
        return record

    def _artifact_links(self, run_id: str, sources: list[dict[str, Any]],
                        depth: int = 0) -> list[str]:
        """The registry entries behind a figure's sources, in order, deduplicated.

        Two shapes occur. A rendered figure records the files it was drawn from,
        each with a path and a digest; a *comparison* records the other figures
        it was built from. Both resolve to artifact ids here, so a caller never
        has to know which kind of figure it is looking at.
        """
        if depth > 3 or not run_id:
            return []
        job_dir = self.settings.job_dir(run_id)
        project = job_dir / "project"
        if not project.is_dir():
            return []
        found: list[str] = []
        for source in sources or []:
            if not isinstance(source, dict):
                continue
            linked = source.get("visualization_id")
            if linked:
                try:
                    nested = self.status(str(linked))
                except (VisualizationError, ValueError):
                    continue
                candidates = nested.get("source_artifact_ids") or self._artifact_links(
                    str(nested.get("run_id") or ""), nested.get("sources") or [], depth + 1)
            else:
                record = ArtifactRegistry(job_dir, run_id, project).by_path_and_hash(
                    str(source.get("relative_path") or ""), source.get("sha256"))
                candidates = [record["artifact_id"]] if record else []
            for identifier in candidates:
                if identifier not in found:
                    found.append(identifier)
        return found

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        root = self.settings.visualizations_root
        if not root.is_dir():
            return []
        rows = []
        for directory in sorted(root.iterdir()):
            record_path = directory / "visualization.json"
            if not directory.is_dir() or not record_path.is_file():
                continue
            try:
                record = read_json(record_path)
            except ValueError:
                continue
            if record.get("run_id") == run_id:
                rows.append(record)
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows

    # ---------------------------------------------------------------- writing
    def create(
        self,
        run_id: str,
        kind: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = catalog.by_kind(kind)
        job_dir = self.settings.job_dir(run_id)
        if not job_dir.is_dir():
            raise VisualizationError(f"任务不存在：{run_id}")
        merged = {**spec.options, **(options or {})}
        visualization_id = new_id(VISUALIZATION_ID_PREFIX)
        directory = self.directory(visualization_id)
        directory.mkdir(parents=True, exist_ok=False)
        record = {
            "visualization_id": visualization_id,
            "run_id": run_id,
            "kind": spec.kind,
            "stage_id": spec.stage_id,
            "title": spec.title,
            "options": merged,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "renderer_version": RENDERER_VERSION,
        }
        write_json(self._record_path(visualization_id), record)
        self._dispatch(visualization_id)
        return record

    def _dispatch(self, visualization_id: str) -> None:
        with self._lock:
            if visualization_id in self._active:
                return
            if len(self._active) >= MAXIMUM_CONCURRENT_VISUALIZATIONS:
                return
            self._active.add(visualization_id)
        thread = threading.Thread(target=self._run, args=(visualization_id,),
                                  name=f"viz-{visualization_id}", daemon=True)
        thread.start()

    def pump(self) -> None:
        """Start queued visualizations while capacity remains."""
        root = self.settings.visualizations_root
        if not root.is_dir():
            return
        for directory in sorted(root.iterdir()):
            record_path = directory / "visualization.json"
            if not record_path.is_file():
                continue
            with self._lock:
                if len(self._active) >= MAXIMUM_CONCURRENT_VISUALIZATIONS:
                    return
            try:
                record = read_json(record_path)
            except ValueError:
                continue
            if record.get("status") == "queued":
                self._dispatch(record["visualization_id"])

    def _run(self, visualization_id: str) -> None:
        record_path = self._record_path(visualization_id)
        try:
            record = read_json(record_path)
            record.update({"status": "running", "updated_at": utc_now()})
            write_json(record_path, record)
            job_dir = self.settings.job_dir(record["run_id"])
            project = job_dir / "project"
            status = read_json(job_dir / "status.json")
            context = RenderContext(job_dir=job_dir, project=project,
                                    run_id=record["run_id"], status=status)
            spec = catalog.by_kind(record["kind"])
            _style()
            renderer = RENDERERS.get(spec.renderer)
            if renderer is None:
                raise VisualizationError(f"未实现渲染器：{spec.renderer}")
            payload = renderer(context, record.get("options") or {})
            figure = payload["figure"]
            _stamp(figure, context, spec.title)
            destination = self.directory(visualization_id) / f"{spec.kind}.png"
            saved = _save(figure, destination)
            sources = []
            for source in payload.get("sources") or []:
                source = Path(source)
                if not source.is_file():
                    continue
                resolved = source.resolve()
                scope = "project"
                try:
                    relative = resolved.relative_to(project.resolve()).as_posix()
                except ValueError:
                    # Job-level inputs such as the QA report live beside the
                    # project, not inside it; record them against the job root.
                    try:
                        relative = resolved.relative_to(job_dir.resolve()).as_posix()
                        scope = "job"
                    except ValueError:
                        continue
                sources.append({"scope": scope, "relative_path": relative,
                                "sha256": sha256_file(source),
                                "size_bytes": source.stat().st_size})
            if not sources:
                record["warnings"] = list(record.get("warnings") or []) + [
                    "该图未绑定任何源文件：对应阶段可能尚未运行"
                ]
            record.update({
                "status": "succeeded",
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "image": destination.name,
                "image_sha256": sha256_file(destination),
                **saved,
                "sources": sources,
                "source_artifact_ids": self._artifact_links(record["run_id"], sources),
                "qa_failed": context.qa_failed,
                "limitations": (
                    ["运行未通过质量门槛；图件带有 QA FAILED 标记，不得作为结论使用"]
                    if context.qa_failed else []
                ),
            })
        except BaseException:  # noqa: BLE001 - a failed figure is recorded, never raised
            try:
                record = read_json(record_path)
            except ValueError:
                record = {"visualization_id": visualization_id}
            record.update({
                "status": "failed",
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "error": traceback.format_exc().strip().splitlines()[-1][:500],
            })
        finally:
            write_json(record_path, record)
            with self._lock:
                self._active.discard(visualization_id)
            self.pump()

    def image_path(self, visualization_id: str) -> Path:
        record = self.status(visualization_id)
        if record.get("status") != "succeeded":
            raise VisualizationError(f"可视化尚未完成：{record.get('status')}")
        path = self.directory(visualization_id) / str(record.get("image"))
        if not path.is_file():
            raise VisualizationError("可视化图件缺失")
        return path

    def restore(self) -> dict[str, Any]:
        """After a restart, nothing is rendering, so requeue what was running."""
        requeued = []
        root = self.settings.visualizations_root
        if not root.is_dir():
            return {"requeued": requeued}
        for directory in sorted(root.iterdir()):
            record_path = directory / "visualization.json"
            if not record_path.is_file():
                continue
            try:
                record = read_json(record_path)
            except ValueError:
                continue
            if record.get("status") in {"queued", "running"}:
                record.update({"status": "queued", "updated_at": utc_now(),
                               "summary": "服务重启后重新排队"})
                write_json(record_path, record)
                requeued.append(record["visualization_id"])
        self.pump()
        return {"requeued": requeued}

    # ------------------------------------------------------------- comparison
    def _source_path(self, record: dict[str, Any], source: dict[str, Any]) -> Path:
        job_dir = self.settings.job_dir(record["run_id"])
        root = job_dir if source.get("scope") == "job" else job_dir / "project"
        root = root.resolve()
        path = (root / str(source["relative_path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise VisualizationError("比较源文件超出任务目录") from exc
        return path

    def _common_rasters(self, records: list[dict[str, Any]]) -> list[Path]:
        source_maps = []
        for record in records:
            mapping = {
                str(item.get("relative_path")): item
                for item in record.get("sources") or []
                if str(item.get("relative_path", "")).lower().endswith((".tif", ".tiff"))
            }
            source_maps.append(mapping)
        common = set(source_maps[0])
        for mapping in source_maps[1:]:
            common.intersection_update(mapping)
        if not common:
            raise VisualizationError("差值和百分比变化仅支持具有同名 GeoTIFF 源的同类图件")
        relative = sorted(common)[0]
        paths = [self._source_path(record, mapping[relative])
                 for record, mapping in zip(records, source_maps)]
        if not all(path.is_file() for path in paths):
            raise VisualizationError("比较所需的 GeoTIFF 源文件缺失")
        return paths

    def _render_comparison(
        self, records: list[dict[str, Any]], mode: str
    ) -> tuple[object, list[str], dict[str, Any] | None]:
        """The figure, any notes, and — for a numeric comparison — the values.

        Returning the values is the point: a difference is data, and a picture
        of it is not. The caller writes them out as a real raster so the
        comparison can be downloaded and recomputed rather than only looked at.
        """
        notes: list[str] = []
        if mode == "side_by_side":
            figure, axes = plt.subplots(1, len(records),
                                        figsize=(5.2 * len(records), 4.8),
                                        constrained_layout=True)
            axes = np.atleast_1d(axes)
            for axis, record in zip(axes, records):
                image_path = self.image_path(record["visualization_id"])
                axis.imshow(plt.imread(image_path))
                axis.set_title(f"{record.get('title') or record['kind']}\n{record['run_id']}",
                               fontsize=8)
                axis.axis("off")
            figure.suptitle("可视化并排比较", fontsize=11)
            return figure, notes, None

        if len(records) != 2:
            raise VisualizationError("差值或百分比变化比较必须且只能选择两个可视化")
        if records[0]["kind"] != records[1]["kind"]:
            raise VisualizationError("差值或百分比变化要求两个可视化类型相同")
        paths = self._common_rasters(records)
        datasets = [rasterio.open(path) for path in paths]
        try:
            first, second = datasets
            if (first.shape != second.shape or first.crs != second.crs
                    or first.transform != second.transform):
                raise VisualizationError("两个栅格的尺寸、CRS 或仿射变换不同，不能逐像元比较")
            before = first.read(1, masked=True).astype("float64")
            after = second.read(1, masked=True).astype("float64")
            mask = np.ma.getmaskarray(before) | np.ma.getmaskarray(after)
            if mode == "difference":
                values = np.ma.array(after - before, mask=mask)
                label = "差值（后 - 前）"
                cmap = "RdBu_r"
            else:
                finite_before = np.abs(before.compressed())
                epsilon = max(float(np.percentile(finite_before, 1)) * 1e-6, 1e-12) \
                    if finite_before.size else 1e-12
                zero_mask = np.abs(before.filled(0.0)) <= epsilon
                mask = mask | zero_mask
                values = np.ma.array(100.0 * (after - before) / np.abs(before), mask=mask)
                label = "变化率 (%)"
                cmap = "RdBu_r"
                excluded = int(np.count_nonzero(zero_mask))
                if excluded:
                    notes.append(f"百分比变化中排除了 {excluded} 个基准值接近零的像元")
            finite = values.compressed()
            if not finite.size:
                raise VisualizationError("两个栅格没有可比较的有效像元")
            limit = float(np.percentile(np.abs(finite), 98))
            limit = limit if limit > 0 else 1.0
            bounds = first.bounds
            crs = first.crs
            derived = {
                "values": values,
                "transform": first.transform,
                "crs": crs,
                "nodata": (first.nodata if first.nodata is not None else -9999.0),
                "label": label,
                "mode": mode,
                "source_paths": [str(path) for path in paths],
            }
        finally:
            for dataset in datasets:
                dataset.close()
        figure, axis = plt.subplots(figsize=(8.0, 6.2), constrained_layout=True)
        image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, origin="upper",
                            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        figure.colorbar(image, ax=axis, shrink=0.82, label=label)
        axis.set_title(f"{records[1]['run_id']} 相对 {records[0]['run_id']}\n{label}")
        axis.set_xlabel("投影坐标 X")
        axis.set_ylabel("投影坐标 Y")
        axis.text(0.01, 0.01, f"CRS: {crs or '未知'}", transform=axis.transAxes,
                  fontsize=6, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
        return figure, notes, derived

    def _write_comparison_raster(
        self, run_id: str, comparison_id: str, derived: dict[str, Any],
        source_artifact_ids: list[list[str]],
    ) -> dict[str, Any] | None:
        """Write the compared values as a real GeoTIFF and register them.

        The figure is a picture of the comparison; this is the comparison. It
        keeps the CRS, transform and NoData of the rasters it came from, records
        which artifacts it was computed from and by what rule, and is written to
        a temporary name first so a half-written raster is never registered or
        served.
        """
        job_dir = self.settings.job_dir(run_id)
        project = job_dir / "project"
        if not project.is_dir():
            return None
        target = (project / "comparison_results"
                  / f"comparison_{derived['mode']}_{comparison_id}.tif")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        values = derived["values"]
        nodata = float(derived["nodata"])
        profile = {
            "driver": "GTiff", "height": int(values.shape[0]), "width": int(values.shape[1]),
            "count": 1, "dtype": "float32", "crs": derived["crs"],
            "transform": derived["transform"], "nodata": nodata, "compress": "deflate",
        }
        try:
            with rasterio.open(temporary, "w", **profile) as sink:
                sink.write(values.filled(nodata).astype("float32"), 1)
            replace_with_retry(temporary, target)
        except BaseException:  # noqa: BLE001 - a derived raster is optional
            temporary.unlink(missing_ok=True)
            logger.warning("比较栅格写入失败", exc_info=True)
            return None
        order = [ids[0] if ids else "" for ids in source_artifact_ids]
        derivation = {
            "mode": derived["mode"],
            "formula": ("after - before" if derived["mode"] == "difference"
                        else "100 * (after - before) / abs(before)"),
            "before_artifact_id": order[0] or None,
            "after_artifact_id": (order[1] if len(order) > 1 else None),
            "mask_rule": "任一侧为 NoData 的像元不参与比较，结果写 NoData",
        }
        if derived["mode"] == "percent_change":
            derivation["zero_rule"] = "基准值绝对值接近零的像元被排除，结果写 NoData"
        registry = ArtifactRegistry(job_dir, run_id, project)
        # A difference of two quantities keeps their unit: subtracting metres
        # gives metres, and a safety factor is dimensionless either way.
        units = None
        for identifiers in source_artifact_ids:
            for identifier in identifiers:
                try:
                    source_units = registry.by_id(identifier).get("units")
                except ValueError:
                    continue
                if source_units:
                    units = source_units
                    break
            if units:
                break
        return registry.register(
            target, "comparison", kind="comparison",
            derived_from=[item for item in order if item],
            producer="watershed_hydro_stability",
            producer_version=RENDERER_VERSION,
            derivation=derivation, units=units,
        )

    def compare(self, visualization_ids: list[str], mode: str = "side_by_side") -> dict[str, Any]:
        if len(visualization_ids) < 2:
            raise VisualizationError("比较至少需要两个可视化编号")
        if len(visualization_ids) > 8:
            raise VisualizationError("一次最多比较 8 个可视化")
        if mode not in {"side_by_side", "difference", "percent_change"}:
            raise VisualizationError("mode 必须是 side_by_side、difference 或 percent_change")
        records = [self.status(item) for item in visualization_ids]
        unfinished = [item["visualization_id"] for item in records
                      if item.get("status") != "succeeded"]
        if unfinished:
            raise VisualizationError(f"以下可视化尚未完成：{unfinished}")
        finished = records
        run_ids = {item["run_id"] for item in finished}
        differences = []
        if len(run_ids) > 1:
            differences.append(f"这些图来自不同的运行：{sorted(run_ids)}")
        if len({item["kind"] for item in finished}) > 1:
            differences.append("比较的图件类型不同，只能比较版式与来源，不能比较数值")
        sources = {}
        for item in finished:
            for source in item.get("sources") or []:
                relative = source.get("relative_path")
                digest = source.get("sha256")
                if relative and digest:
                    sources.setdefault(relative, set()).add(digest)
        changed = [path for path, digests in sources.items() if len(digests) > 1]
        if changed:
            differences.append(f"源文件内容不同：{sorted(changed)}")
        _style()
        figure, render_notes, derived = self._render_comparison(finished, mode)
        comparison_id = new_id(VISUALIZATION_ID_PREFIX)
        directory = self.directory(comparison_id)
        directory.mkdir(parents=True, exist_ok=False)
        destination = directory / f"comparison_{mode}.png"
        saved = _save(figure, destination)
        comparison_record = {
            "visualization_id": comparison_id,
            "run_id": finished[0]["run_id"],
            "source_run_ids": sorted(run_ids),
            "kind": f"comparison_{mode}",
            "stage_id": "comparison",
            "title": {"side_by_side": "可视化并排比较", "difference": "栅格差值",
                      "percent_change": "栅格百分比变化"}[mode],
            "mode": mode,
            "source_visualization_ids": visualization_ids,
            "status": "succeeded",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "finished_at": utc_now(),
            "renderer_version": RENDERER_VERSION,
            "image": destination.name,
            "image_sha256": sha256_file(destination),
            "qa_failed": any(item.get("qa_failed") for item in finished),
            "warnings": render_notes,
            "limitations": (["至少一个来源运行未通过质量门槛，比较图不得作为已验证结论"]
                            if any(item.get("qa_failed") for item in finished) else []),
            "sources": [
                {"visualization_id": item["visualization_id"],
                 "image_sha256": item.get("image_sha256")}
                for item in finished
            ],
            **saved,
        }
        if derived is not None:
            # The numeric comparison is written out and registered, so the
            # difference can be downloaded as data and recomputed rather than
            # only looked at. The figure carries the new file as its source, so
            # the ordinary artifact-link path finds it.
            written = self._write_comparison_raster(
                str(comparison_record["run_id"]), comparison_id, derived,
                [list(item.get("source_artifact_ids") or []) for item in finished])
            if written is not None:
                comparison_record["sources"] = list(comparison_record["sources"]) + [{
                    "scope": "project", "relative_path": written["internal_path"],
                    "sha256": written["sha256"], "size_bytes": written["size_bytes"]}]
                comparison_record["source_artifact_ids"] = [written["artifact_id"]]
                comparison_record["derivation"] = written.get("derivation")
            else:
                comparison_record["warnings"] = list(comparison_record["warnings"]) + [
                    "数值比较结果未能写盘，只能查看图件，不能下载差值栅格"]
        write_json(self._record_path(comparison_id), comparison_record)
        return {
            "comparison_visualization_id": comparison_id,
            "mode": mode,
            "visualization_ids": [item["visualization_id"] for item in finished],
            "run_ids": sorted(run_ids),
            "kinds": sorted({item["kind"] for item in finished}),
            "same_sources": not changed,
            "differences": differences,
            "qa_failed_count": sum(1 for item in finished if item.get("qa_failed")),
            "comparison": comparison_record,
            "entries": [
                {
                    "visualization_id": item["visualization_id"],
                    "run_id": item["run_id"],
                    "kind": item["kind"],
                    "title": item.get("title"),
                    "created_at": item.get("created_at"),
                    "qa_failed": bool(item.get("qa_failed")),
                    "source_count": len(item.get("sources") or []),
                    "image_sha256": item.get("image_sha256"),
                }
                for item in finished
            ],
        }


SERVICE = VisualizationService(SETTINGS)
