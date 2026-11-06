"""Deletion plans for explicitly identified, service-owned products.

Never accepts filesystem paths from a caller. Dependencies protect products
still used outside the deletion set. Missing products make retries idempotent.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PathSafetyError


PRODUCT_ID = re.compile(r"^(?:basin-|rain-|package-|hydro-input-|hydro-run-|viz-)[0-9a-f]{12}$")
REFERENCE_KEYS = {
    "review_id", "rainfall_package_id", "model_input_package_id", "package_id",
    "source_package_id", "hydro_input_id", "input_id", "hydro_run_id", "run_id",
    "visualization_id", "comparison_visualization_id", "source_run_ids",
    "source_visualization_ids", "visualization_ids",
}


def product_ids(values) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("产物编号必须是列表")
    if any(not isinstance(value, str) or not PRODUCT_ID.fullmatch(value) for value in values):
        raise ValueError("产物编号无效；不接受路径或任意目录名")
    return set(values)


def references(value) -> set[str]:
    """Read identifiers from structured metadata fields, never free-form prose."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in REFERENCE_KEYS:
                candidates = item if isinstance(item, list) else [item]
                found.update(x for x in candidates if isinstance(x, str) and PRODUCT_ID.fullmatch(x))
            if isinstance(item, (dict, list)):
                found.update(references(item))
    elif isinstance(value, list):
        for item in value:
            found.update(references(item))
    return found


def linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def safe_target(root: Path, target: Path, *, inspect_tree: bool = True) -> None:
    """Check absolute containment and reject links before any recursive delete."""
    root, target = Path(root).absolute(), Path(target).absolute()
    if target.parent != root or any(linked(p) for p in (root, *root.parents, target)):
        raise PathSafetyError("清理路径含链接或不在指定受管目录内")
    if target.resolve().parent != root.resolve():
        raise PathSafetyError("清理路径超出受管目录")
    if inspect_tree and target.is_dir():
        # rglob does not descend symlinks; the link itself is still rejected.
        for child in target.rglob("*"):
            if linked(child):
                raise PathSafetyError("产物目录含链接，拒绝递归删除")


def metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    if linked(path):
        raise PathSafetyError("产物清单含链接")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("产物清单无法读取；为避免误删，请修复清单后重试") from exc
    if not isinstance(value, dict):
        raise ValueError("产物清单格式无效")
    return value


@dataclass
class Product:
    identifier: str
    targets: list[tuple[Path, Path]]
    dependencies: set[str] = field(default_factory=set)
    active: bool = False


def catalog_directories(root: Path, prefix: str, filename: str) -> dict[str, Product]:
    result = {}
    if not root.exists():
        return result
    for path in root.iterdir():
        if not PRODUCT_ID.fullmatch(path.name) or not path.name.startswith(prefix):
            continue
        safe_target(root, path, inspect_tree=False)
        if not path.is_dir():
            raise PathSafetyError("受管产物应为目录")
        record = metadata(path / filename)
        result[path.name] = Product(path.name, [(root, path)], references(record) - {path.name},
                                    record.get("status") in {"queued", "running"})
    return result


def closure(seeds: set[str], catalog: dict[str, Product]) -> set[str]:
    found = set(seeds)
    pending = list(seeds)
    while pending:
        item = catalog.get(pending.pop())
        for dependency in item.dependencies if item else ():
            if dependency not in found:
                found.add(dependency)
                pending.append(dependency)
    return found


def cleanup_products(catalog: dict[str, Product], identifiers: list[str],
                     protected_ids: list[str], *, dry_run: bool) -> dict:
    seeds, protected = product_ids(identifiers), product_ids(protected_ids)
    selected = closure(seeds, catalog)
    # All figures belonging to deleted runs are products of those runs. A
    # cross-run comparison belongs here only when all its runs are selected.
    for identifier, item in catalog.items():
        run_dependencies = {x for x in item.dependencies if x.startswith("hydro-run-")}
        if identifier.startswith("viz-") and run_dependencies and run_dependencies <= selected:
            selected.add(identifier)
    selected = closure(selected, catalog)
    retained = closure(protected | (set(catalog) - selected), catalog)
    # A retained model remains a usable bundle, including all of its figures.
    for identifier, item in catalog.items():
        if identifier.startswith("viz-") and any(
            x.startswith("hydro-run-") and x in retained for x in item.dependencies
        ):
            retained.add(identifier)
    retained = closure(retained, catalog)
    deletion = (selected & set(catalog)) - retained
    if any(catalog[x].active for x in deletion):
        raise RuntimeError("该对话仍有计算或图件任务运行；请先取消任务，确认停止后再删除。")
    # Validate every target before making the first destructive change.
    for identifier in sorted(deletion):
        for root, path in catalog[identifier].targets:
            safe_target(root, path)
    order = []
    pending = set(deletion)
    while pending:
        ready = sorted(x for x in pending if not any(
            x in catalog[y].dependencies for y in pending if y != x))
        if not ready:
            raise ValueError("产物依赖出现循环，拒绝清理")
        order.extend(ready)
        pending.difference_update(ready)
    deleted = []
    if not dry_run:
        # Delete dependents first; a partially completed retry can still find
        # and protect the dependencies of any remaining products.
        for identifier in order:
            for root, path in catalog[identifier].targets:
                safe_target(root, path)
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            deleted.append(identifier)
    return {
        "dry_run": dry_run, "planned_ids": sorted(deletion), "deleted_ids": deleted,
        "retained_ids": sorted(selected & retained), "missing_ids": sorted(seeds - set(catalog)),
        "external_references": sorted(retained - set(catalog)),
        "related_external_ids": sorted(selected - set(catalog)),
    }
