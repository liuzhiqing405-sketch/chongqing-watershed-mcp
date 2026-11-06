"""Approximate GCJ-02/WGS84 conversion for DataV reference boundaries.

Forward formula adapted from wandergis/coordtransform (MIT); license in
THIRD_PARTY_NOTICES.md. Inversion tolerance describes numerical convergence,
not the geographic accuracy of a provider's boundary.
"""
import numpy as np


def wgs84_to_gcj02(longitude, latitude):
    lon,lat=np.asarray(longitude,dtype=float),np.asarray(latitude,dtype=float)
    x,y=lon-105.,lat-35.
    pi=np.pi
    common=(20*np.sin(6*x*pi)+20*np.sin(2*x*pi))*2/3
    dy=(-100+2*x+3*y+.2*y*y+.1*x*y+.2*np.sqrt(np.abs(x))+common
        +(20*np.sin(y*pi)+40*np.sin(y/3*pi))*2/3
        +(160*np.sin(y/12*pi)+320*np.sin(y*pi/30))*2/3)
    dx=(300+x+2*y+.1*x*x+.1*x*y+.1*np.sqrt(np.abs(x))+common
        +(20*np.sin(x*pi)+40*np.sin(x/3*pi))*2/3
        +(150*np.sin(x/12*pi)+300*np.sin(x/30*pi))*2/3)
    rad=np.radians(lat);magic=1-.00669342162296594323*np.sin(rad)**2
    dy=dy*180/((6378245.*(1-.00669342162296594323))/(magic*np.sqrt(magic))*pi)
    dx=dx*180/(6378245./np.sqrt(magic)*np.cos(rad)*pi)
    inside=(lon>73.66)&(lon<135.05)&(lat>3.86)&(lat<53.55)
    return lon+np.where(inside,dx,0),lat+np.where(inside,dy,0)


def gcj02_to_wgs84(longitude, latitude):
    target_x,target_y=np.asarray(longitude,dtype=float),np.asarray(latitude,dtype=float)
    x,y=target_x.copy(),target_y.copy()
    for _ in range(10):
        mapped_x,mapped_y=wgs84_to_gcj02(x,y)
        dx,dy=mapped_x-target_x,mapped_y-target_y
        x,y=x-dx,y-dy
        if max(np.max(np.abs(dx)),np.max(np.abs(dy)))<1e-9:
            return x,y
    raise ValueError("GCJ-02 边界坐标转换未收敛")
