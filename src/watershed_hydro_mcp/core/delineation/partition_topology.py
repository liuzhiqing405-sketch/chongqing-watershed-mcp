"""Coarsen D8 catchments along their drainage graph, never across divides."""
from __future__ import annotations

import heapq
import numpy as np


def coarsen_labels(labels, receivers, valid, *, minimum_cells, target_count):
    grid = np.asarray(labels, dtype=np.int32)
    flat = grid.ravel()
    counts = np.bincount(flat[valid.ravel()])
    n = len(counts)
    parent = np.arange(n, dtype=np.int32)
    downstream = np.zeros(n, dtype=np.int32)
    upstream = [set() for _ in range(n)]
    active = set(map(int, np.flatnonzero(counts[1:]) + 1))
    ids = np.flatnonzero(valid.ravel() & (receivers >= 0))
    pairs = np.column_stack((flat[ids], flat[receivers[ids]]))
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    targets = {}
    for source, target in pairs:
        targets.setdefault(int(source), set()).add(int(target))
    for source, choices in targets.items():
        if len(choices) != 1 or 0 in choices:
            raise RuntimeError(f"Partition {source} has conflicting downstream catchments.")
        target = next(iter(choices))
        downstream[source] = target
        upstream[target].add(source)

    def merge(source, target):
        parent[source] = target
        counts[target] += counts[source]
        counts[source] = 0
        active.remove(source)
        upstream[target].discard(source)
        for donor in upstream[source]:
            downstream[donor] = target
            upstream[target].add(donor)
        upstream[source].clear()
        downstream[source] = 0

    # Exhaust the actual graph, without an arbitrary 2000-iteration cutoff.
    heap = [(int(counts[k]), k) for k in active if counts[k] < minimum_cells]
    heapq.heapify(heap)
    small_merges = 0
    while heap:
        size, label = heapq.heappop(heap)
        if label not in active or counts[label] != size or size >= minimum_cells:
            continue
        target = int(downstream[label])
        if target:
            merge(label, target)
        elif upstream[label]:
            source = min(upstream[label], key=lambda k: (counts[k], k))
            merge(source, label)
            target = label
        else:
            continue  # A genuine independent boundary outlet; retain it.
        small_merges += 1
        if counts[target] < minimum_cells:
            heapq.heappush(heap, (int(counts[target]), target))

    # Reduce internal river reaches only. Independent outlets cannot be joined
    # just to achieve a requested partition count.
    heap = [(int(counts[k]), k) for k in active if downstream[k]]
    heapq.heapify(heap)
    budget_merges = 0
    def major_count():
        return sum(counts[k] >= minimum_cells for k in active)
    while major_count() > target_count and heap:
        size, label = heapq.heappop(heap)
        if label not in active or counts[label] != size or not downstream[label]:
            continue
        target = int(downstream[label])
        merge(label, target)
        budget_merges += 1
        if downstream[target]:
            heapq.heappush(heap, (int(counts[target]), target))

    lookup = np.zeros(n, dtype=np.int32)
    for new, label in enumerate(sorted(active), 1):
        lookup[label] = new
    for label in range(1, n):
        trail = []
        root = label
        while parent[root] != root:
            trail.append(root)
            root = int(parent[root])
        for k in trail:
            parent[k] = root
        lookup[label] = lookup[root]
    out = lookup[grid]
    terminal = {int(lookup[k]) for k in active if not downstream[k]}
    facts = {"small_merges": small_merges, "budget_merges": budget_merges,
             "partition_count": len(active), "target_count": int(target_count),
             "target_reached": len(active) <= target_count,
             "major_partition_count": int(major_count()),
             "major_target_reached": bool(major_count() <= target_count),
             "independent_outlet_count": len(terminal),
             "retained_small_terminal_count": int(sum(counts[k] < minimum_cells for k in active if not downstream[k])),
             "retained_small_terminal_cells": int(sum(counts[k] for k in active if not downstream[k] and counts[k] < minimum_cells))}
    return out, terminal, facts


def partition_facts(labels, receivers, valid, boundary):
    """Check full coverage, nonempty labels and actual domain terminal outlets."""
    flat = labels.ravel()
    valid_flat = valid.ravel()
    terminal_ids = np.flatnonzero(valid_flat & (receivers < 0))
    facts = {"valid_cells": int(valid.sum()),
             "labelled_cells": int(np.count_nonzero(valid & (labels > 0))),
             "labels_outside_domain": int(np.count_nonzero(~valid & (labels > 0))),
             "internal_terminal_cells": int(np.count_nonzero(~boundary.ravel()[terminal_ids]))}
    counts = np.bincount(flat[valid_flat])
    facts["single_cell_partition_count"] = int(np.count_nonzero(counts[1:] == 1))
    facts["single_cell_area_fraction"] = float(np.count_nonzero(counts[1:] == 1) / max(1, valid.sum()))
    facts["passed"] = bool(facts["labelled_cells"] == facts["valid_cells"]
                           and not facts["labels_outside_domain"]
                           and not facts["internal_terminal_cells"])
    return facts


def single_exit_facts(labels, receivers, valid):
    """Require one first exit per basin, including diagonal D8 connections.

    In a separately checked acyclic SFD graph this proves that all cells of
    each label converge to its exit. Multiple exits to even the same downstream
    label do not satisfy the common-outlet requirement.
    """
    flat = labels.ravel()
    ids = np.flatnonzero(valid.ravel() & (flat > 0))
    basins = np.unique(flat[ids])
    downstream = receivers[ids]
    exits = ids[(downstream < 0) | (flat[np.maximum(downstream, 0)] != flat[ids])]
    counts = np.bincount(flat[exits], minlength=int(basins.max())+1 if len(basins) else 1)
    bad = basins[counts[basins] != 1]
    return {
        "partition_count": int(len(basins)),
        "exit_cell_count": int(len(exits)),
        "single_exit_partition_count": int(np.count_nonzero(counts[basins] == 1)),
        "missing_exit_partition_count": int(np.count_nonzero(counts[basins] == 0)),
        "multiple_exit_partition_count": int(np.count_nonzero(counts[basins] > 1)),
        "maximum_exits_per_partition": int(counts.max()),
        "invalid_partition_count": int(len(bad)),
        "invalid_partitions": [{"label": int(k), "exit_cells": int(counts[k])} for k in bad[:20]],
        "passed": bool(len(basins) and not len(bad)),
    }


def clip_labels_to_local_drainage(labels, receivers, valid):
    """Split buffered identities at local exits; never connect across outside land."""
    flat = labels.ravel()
    ids = np.flatnonzero(valid.ravel())
    seeds = np.zeros(flat.size, dtype=np.int32)
    exits = ids[(receivers[ids] < 0) | (flat[np.maximum(receivers[ids], 0)] != flat[ids])]
    seeds[exits] = np.arange(1, len(exits)+1)
    for start in ids:
        if seeds[start]:
            continue
        path, cur = [], int(start)
        while cur >= 0 and not seeds[cur]:
            path.append(cur)
            cur = int(receivers[cur])
        if cur < 0:
            raise RuntimeError("Buffered partition has an unseeded local outlet")
        seeds[path] = seeds[cur]
    return seeds.reshape(labels.shape)


def filter_boundary_fragments(labels, receivers, valid, *, cell_area_ha, minimum_area_ha, maximum_fraction):
    """Drop only small independent boundary trees, with explicit area accounting."""
    from .delineate import domain_boundary
    if not np.isfinite(minimum_area_ha) or minimum_area_ha < 0 or not 0 <= maximum_fraction <= 1:
        raise ValueError("Invalid boundary fragment filtering policy")
    counts = np.bincount(labels.ravel())
    flat = labels.ravel()
    ids = np.flatnonzero(valid.ravel())
    boundary_labels = set(map(int, np.unique(labels[domain_boundary(valid)]))) - {0}
    terminals = set(map(int, flat[ids[receivers[ids] < 0]]))
    crossings = ids[(receivers[ids] >= 0) & (flat[np.maximum(receivers[ids], 0)] != flat[ids])]
    # Neither a downstream link nor an upstream donor may be severed.
    connected = set(map(int, flat[crossings])) | set(map(int, flat[receivers[crossings]]))
    removed = sorted(k for k in (boundary_labels & terminals)-connected
                     if counts[k]*cell_area_ha < minimum_area_ha)
    excluded = np.isin(labels, removed) & valid
    fraction = float(excluded.sum()/max(1, valid.sum()))
    if fraction > maximum_fraction + 1e-12:
        raise RuntimeError(f"Boundary fragments would exclude {fraction:.2%}, above {maximum_fraction:.2%}; adjust the explicit policy or boundary")
    retained = valid & ~excluded
    if not retained.any():
        raise RuntimeError("Boundary filtering would remove the entire calculation domain")
    out = np.where(retained, labels, 0).astype(np.int32)
    records = [{"label": k, "cells": int(counts[k]), "area_ha": float(counts[k]*cell_area_ha),
                "reason": "independent_boundary_outlet_below_minimum_area"} for k in removed]
    facts = {"minimum_area_ha": float(minimum_area_ha), "maximum_excluded_fraction": float(maximum_fraction),
             "original_cells": int(valid.sum()), "retained_cells": int(retained.sum()),
             "excluded_cells": int(excluded.sum()), "excluded_count": len(removed), "excluded_fraction": fraction,
             "original_area_ha": float(valid.sum()*cell_area_ha),
             "retained_area_ha": float(retained.sum()*cell_area_ha),
             "excluded_area_ha": float(excluded.sum()*cell_area_ha), "excluded_units": records}
    return out, retained, excluded, facts
