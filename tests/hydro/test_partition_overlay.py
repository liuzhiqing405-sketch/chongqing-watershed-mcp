from pyproj import CRS, Transformer
import pytest
from watershed_hydro_mcp.visualization.renderer import _rings_in_crs


def test_reference_river_longitude_latitude_reprojects_to_dem_metres(tmp_path):
    path=tmp_path/'river.shp';path.with_suffix('.prj').write_text(CRS.from_epsg(4326).to_wkt(),encoding='utf-8')
    target=CRS.from_epsg(32649)
    ring=[(109.5,31.4),(109.6,31.5)]
    projected=_rings_in_crs([ring],path,target)[0]
    expected=Transformer.from_crs(4326,target,always_xy=True).transform(109.5,31.4)
    assert projected[0]==pytest.approx(expected)
    assert projected[0][1]>3_000_000


def test_unknown_vector_crs_is_not_silently_plotted_in_dem_coordinates(tmp_path):
    with pytest.raises(ValueError,match='坐标系'):
        _rings_in_crs([[(109.5,31.4)]],tmp_path/'river.shp',32649)
