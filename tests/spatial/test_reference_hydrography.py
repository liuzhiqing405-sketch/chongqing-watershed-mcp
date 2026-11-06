"""Reference lines must stay aligned with the selection map's click grid."""
import numpy as np
import pytest
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from chongqing_spatial_mcp.map_hydrography import segments, _visible_edges
from chongqing_spatial_mcp.settings import Settings


def test_projected_multipart_rivers_remain_separate_and_align_with_wgs84(tmp_path):
    path=tmp_path/'rivers.shp'
    lines=[[(105.8,29.2),(105.85,29.24),(105.9,29.25)],
           [(105.92,29.3),(105.95,29.35)]]
    tx=Transformer.from_crs(4326,32649,always_xy=True)
    with shapefile.Writer(str(path),shapeType=shapefile.POLYLINE) as writer:
        writer.field('HYRIV_ID','N')
        writer.line([[tx.transform(*p) for p in line] for line in lines])
        writer.record(123)
    path.with_suffix('.prj').write_text(CRS.from_epsg(32649).to_wkt(),encoding='utf-8')
    result=segments(path,(105.7,29.1,106,29.5))
    assert len(result)==2
    for actual,expected in zip(result,lines):
        assert LineString(actual).hausdorff_distance(LineString(expected))<1e-8
    path.with_suffix('.prj').unlink()
    with pytest.raises(ValueError,match='坐标系'):
        segments(path,(105.7,29.1,106,29.5))


def test_crossing_edge_is_kept_when_both_endpoints_are_outside():
    points=np.array([[-1,.5],[2,.5],[3,.5]])
    visible=list(_visible_edges(points,(0,0,1,1)))
    assert len(visible)==1
    assert np.array_equal(visible[0],points[:2])


def test_reference_network_does_not_replace_named_calculation_layer(tmp_path):
    base=dict(root=tmp_path,data_root=tmp_path,runs_root=tmp_path,
              exchange_root=tmp_path,native_root=tmp_path)
    settings=Settings(**base,defaults={'datasets':{'rivers':'named.shp','display_rivers':'reference.shp'}})
    assert settings.rivers==tmp_path/'named.shp'
    assert settings.display_rivers==tmp_path/'reference.shp'
    assert Settings(**base,defaults={'datasets':{'rivers':'named.shp'}}).display_rivers is None
