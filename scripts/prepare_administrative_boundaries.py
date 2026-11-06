"""Reproducibly convert the retained DataV boundaries to WGS84 (no download)."""
import hashlib
import json
from pathlib import Path

from shapely import make_valid
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union

from chongqing_spatial_mcp.coordinate_conversion import gcj02_to_wgs84


def main():
    root=Path(__file__).resolve().parents[1]
    raw=root/"data/administrative_districts.geojson"
    data=json.loads(raw.read_text(encoding="utf-8"))
    repairs=[]
    for feature in data["features"]:
        original=shape(feature["geometry"])
        converted=transform(gcj02_to_wgs84,original)
        if not converted.is_valid:
            repaired=make_valid(converted)
            discarded_lines=0
            if repaired.geom_type=="GeometryCollection":
                discarded_lines=sum(g.geom_type not in {"Polygon","MultiPolygon"} for g in repaired.geoms)
                repaired=unary_union([g for g in repaired.geoms if g.geom_type in {"Polygon","MultiPolygon"}])
            if repaired.geom_type not in {"Polygon","MultiPolygon"}:
                raise ValueError(f"Cannot preserve polygon topology: {feature['properties']['name']}")
            repairs.append({"name":feature["properties"]["name"],
                            "discarded_zero_area_parts":discarded_lines,
                            "area_change_deg2":repaired.area-converted.area})
            converted=repaired
        feature["geometry"]=mapping(converted)
        for key in ("center","centroid"):
            if key in feature["properties"]:
                x,y=gcj02_to_wgs84(*feature["properties"][key])
                feature["properties"][key]=[float(x),float(y)]
    data.pop("crs",None)
    data["coordinate_system"]="EPSG:4326"
    output=root/"data/administrative_districts.wgs84.geojson"
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    provenance={
        "source_url":"https://geo.datav.aliyun.com/areas_v3/bound/500000_full.json",
        "provider":"Aliyun DataV GeoAtlas", "source_retrieved_at":"2026-09-30",
        "prepared_at":"2026-10-01", "source_coordinate_system":"GCJ-02",
        "coordinate_system":"EPSG:4326", "raw_file":raw.name,
        "raw_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),
        "converted_file":output.name,
        "converted_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "coordinate_system_evidence":"https://help.aliyun.com/zh/datav/datav-7-0/user-guide/map-data-format-1",
        "conversion":"Numerical inverse of coordtransform WGS84-to-GCJ02, tolerance 1e-9 degree; approximate geographic conversion, not surveyed accuracy.",
        "conversion_source":"https://github.com/wandergis/coordtransform",
        "geometry_repairs":repairs, "feature_count":len(data["features"]),
        "usage":"Administrative DEM map/cutline only. Modeling uses the confirmed user polygon and original full DEM.",
        "accuracy_note":"Public cartographic reference boundaries, not authoritative cadastral or surveyed engineering boundaries. Source detail limits cutline accuracy."
    }
    (root/"data/administrative_districts.source.json").write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"features":len(data["features"]),"repairs":repairs},ensure_ascii=True))


if __name__=="__main__":
    main()
