"""Hydro cleanup with active-job and shared-input protection."""
from watershed_exchange.cleanup import catalog_directories, cleanup_products


def cleanup_hydro_products(identifiers, protected_ids, *, dry_run, store, visualizations):
    settings = store.settings
    with store._lock, visualizations._lock:
        catalog = {}
        catalog.update(catalog_directories(settings.inputs_root, "hydro-input-", "input_record.json"))
        jobs = catalog_directories(settings.jobs_root, "hydro-run-", "job.json")
        for identifier, item in jobs.items():
            status = store.status(identifier)
            item.active = (status.get("status") in {"queued", "running"}
                           or identifier in store._processes)
        catalog.update(jobs)
        figures = catalog_directories(settings.visualizations_root, "viz-", "visualization.json")
        for identifier, item in figures.items():
            item.active = item.active or identifier in visualizations._active
        catalog.update(figures)
        return cleanup_products(catalog, identifiers, protected_ids, dry_run=dry_run)
