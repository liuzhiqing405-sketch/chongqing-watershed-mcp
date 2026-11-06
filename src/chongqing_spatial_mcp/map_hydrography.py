"""Draw an optional reference river network on the WGS84 selection grid."""
from pathlib import Path

import numpy as np
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import LineString


def _visible_edges(points, bounds):
    if len(points) < 2:
        return
    west,south,east,north=bounds
    a,b=points[:-1],points[1:]
    visible=((np.minimum(a[:,0],b[:,0])<=east)&(np.maximum(a[:,0],b[:,0])>=west)
             &(np.minimum(a[:,1],b[:,1])<=north)&(np.maximum(a[:,1],b[:,1])>=south))
    changes=np.diff(np.r_[False,visible,False].astype("int8"))
    for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        yield points[start:end+1]


def segments(path: Path, extent, pixel_width=1200, pixel_height=1000):
    """Read only visible edges; preserve multipart lines and polygon holes.

    Polygons are reference outlines, not measured water surface widths. Display
    simplification is below a quarter pixel and never changes source files.
    """
    prj=path.with_suffix(".prj")
    if not prj.is_file():
        raise ValueError("参考河网缺少 .prj 坐标系，不能可靠叠加到地图")
    crs=CRS.from_user_input(prj.read_text(encoding="utf-8"))
    west,south,east,north=extent
    to_source=Transformer.from_crs(4326,crs,always_xy=True)
    bounds=to_source.transform_bounds(west,south,east,north,densify_pts=21)
    to_map=Transformer.from_crs(crs,4326,always_xy=True)
    tolerance=min((east-west)/pixel_width,(north-south)/pixel_height)*.25
    result=[]
    with shapefile.Reader(str(path)) as reader:
        for item in reader.iterShapes(bbox=bounds):
            if item.shapeType==0:
                continue
            if item.shapeType not in {3,5,13,15,23,25}:
                raise ValueError("参考河网必须是线或面要素")
            points=np.asarray(item.points,dtype="float64")[:,:2]
            parts=list(item.parts)+[len(points)]
            for start,end in zip(parts[:-1],parts[1:]):
                for visible in _visible_edges(points[start:end],bounds):
                    xs,ys=to_map.transform(visible[:,0],visible[:,1])
                    line=LineString(np.column_stack((xs,ys))).simplify(tolerance)
                    result.append(np.asarray(line.coords))
    return result


def draw_reference(axis, path: Path, extent, label: str, pixel_width=1200, pixel_height=1000):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib import patheffects

    lines=segments(path,extent,pixel_width,pixel_height)
    color="#c600a0"
    axis.add_collection(LineCollection(lines,colors="white",linewidths=1.6,
                                      alpha=.9,zorder=4,capstyle="round"))
    axis.add_collection(LineCollection(lines,colors=color,linewidths=.65,
                                      zorder=4.1,capstyle="round"))
    return Line2D([],[],color=color,linewidth=1.2,label=label,
                  path_effects=[patheffects.Stroke(linewidth=2.5,foreground="white"),
                                patheffects.Normal()])
