"""Remove selected spatial products; source layers and map caches are retained."""
from watershed_exchange.cleanup import PRODUCT_ID, Product, catalog_directories, cleanup_products, safe_target

from .settings import SETTINGS


def cleanup_spatial_products(identifiers, protected_ids, *, dry_run=True, settings=SETTINGS):
    catalog = {}
    for root, prefix, manifest in (
        (settings.watersheds_root, "basin-", "review.json"),
        (settings.rainfall_runs_root, "rain-", "rainfall_manifest.json"),
        (settings.model_inputs_root, "package-", "input_manifest.json"),
    ):
        catalog.update(catalog_directories(root, prefix, manifest))
    # Process figures are stored by review, not under a separately owned id.
    catalog.update({key: value for key, value in catalog_directories(
        settings.spatial_visualizations_root, "basin-", "cleanup.json").items() if key not in catalog})
    for identifier, item in list(catalog.items()):
        if identifier.startswith("basin-"):
            target = (settings.spatial_visualizations_root, settings.spatial_visualizations_root / identifier)
            if target not in item.targets:
                item.targets.append(target)
    # Exports may survive a partially completed earlier deletion.
    root = settings.exports_root
    if root.exists():
        for path in root.iterdir():
            identifier = path.name[:-4] if path.suffix == ".zip" else path.name
            if not identifier.startswith("package-") or not PRODUCT_ID.fullmatch(identifier):
                continue
            safe_target(root, path, inspect_tree=False)
            item = catalog.setdefault(identifier, Product(identifier, []))
            item.targets.append((root, path))
    return cleanup_products(catalog, identifiers, protected_ids, dry_run=dry_run)
