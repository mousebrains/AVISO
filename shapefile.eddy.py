#! /usr/bin/env python3
#
# Read in Cyclone and Anticyclone AVISO+ Mesoscale Eddy Product NetCDF files
# and produce shapefiles of the velocity contours for the eddies.
#
# Jan-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString
import re
import time

def wrapTo360(theta:np.array) -> np.array:
    ''' Wrap all values to [0,360) '''
    theta = np.mod(theta, 360)
    if isinstance(theta, (list, tuple, np.ndarray)):
        theta[theta < 0] += 360
    elif theta < 0:
        theta += 360
    return theta

def wrapTo180(theta:np.array) -> np.array:
    ''' Wrap all values onto the interval [-180,180) '''
    theta = np.mod(np.array(theta) + 180, 360)
    if isinstance(theta, (list, tuple, np.ndarray)):
        theta[theta < 0] += 360
    elif theta < 0: 
        theta += 360
    return theta - 180

def fetchContour(ds:xr.Dataset, row:pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if row.oFlag == 0: # This is an observation
        iObs = row.indices
        lat = row.lat
        lon = row.lon
    else: # This is not an observation, so find the previous one
        indices = row.indices + np.arange(-20, 0) # Indices to look for an observation
        trk = ds.track[indices].to_numpy()
        indices = indices[trk == row.trk]
        oFlag = ds.observation_flag[indices].to_numpy()
        iObs = np.nonzero(oFlag == 0)[0]
        if len(iObs) == 0: return (None, None, None) # No prior observations found
        iObs = indices[iObs.max()]
        lat = ds.latitude[iObs].to_numpy()
        lon = ds.longitude[iObs].to_numpy()
    return (
            ds.effective_contour_latitude [iObs].to_numpy() - lat,
            ds.effective_contour_longitude[iObs].to_numpy() - lon,
            ds.uavg_profile[iObs,:].to_numpy(),
            )

def mkContours(ds:xr.Dataset, info:dict, norm:float, tit:str) -> gpd.GeoDataFrame:
    q = np.logical_and.reduce((
        ds.time == info["date"],
        ds.latitude  >= info["latMin"] - 2, # Extra for the contours at the edge
        ds.latitude  <= info["latMax"] + 2,
        ds.longitude >= wrapTo360(info["lonMin"]) - 2, # Extra for the contours at the edge
        ds.longitude <= wrapTo360(info["lonMax"]) + 2,
        ))
    df = pd.DataFrame() # The collection of tracks for this date/lat/lon box
    df["t"] = ds.time[q]
    df["trk"] = ds.track[q]
    df["indices"] = np.nonzero(q)[0]
    df["oFlag"] = ds.observation_flag[q]
    df["lat"] = ds.latitude[q]
    df["lon"] = ds.longitude[q]
    nAvg = ds.uavg_profile.shape[1] # Number of contours per eddy
    scaling = np.arange(nAvg, 0, -1) / nAvg # contour scaling
    gdf = gpd.GeoDataFrame()
    for index, row in df.iterrows(): # Walk through the tracks
        (cLat, cLon, uAvg) = fetchContour(ds, row) # Get the lat/lon centered outer contour
        lon = wrapTo180(row.lon) # For display purposes
        for i in range(nAvg):
            ccLon = cLon * scaling[i] + lon
            ccLat = cLat * scaling[i] + row.lat
            gdf = gdf.append(gpd.GeoDataFrame(data={
                    "trk": (row.trk,),
                    "uAvg": (norm * uAvg[i],), 
                    "contour": (i,),
                    "geometry": (LineString(np.array([ccLon, ccLat]).T),),
                    },
                    crs = "EPSG:4326",
                    ))
    return gdf

parser = ArgumentParser()
grp = parser.add_argument_group(description="Input NetCDF file options")
grp.add_argument("--cyclone", type=str, required=True, help="Cyclone NetCDF data")
grp.add_argument("--anticyclone", type=str, required=True, help="Anticyclone NetCDF data")

grp = parser.add_argument_group(description="Output shapefile options")
grp.add_argument("--fnCyclone", type=str, default="cyclone.shp",
        help="Shapefile filename for the cyclone data")
grp.add_argument("--fnAnticyclone", type=str, default="anticyclone.shp",
        help="Shapefile filename for the anticyclone data")

grp = parser.add_argument_group(description="Data selection criteria options")
grp.add_argument("--date", type=str, default="2019-03-01", help="Date to plot data for, YYYY-MM-DD")
grp.add_argument("--latMin", type=float, default=5, help="Minimum latitude, decimal degrees")
grp.add_argument("--latMax", type=float, default=30, help="Maximum latitude, decimal degrees")
grp.add_argument("--lonMin", type=float, default=130, help="Minimum longitude, decimal degrees")
grp.add_argument("--lonMax", type=float, default=155, help="Maximum longitude, decimal degrees")

args = parser.parse_args()

if not re.match("^\d{4}-\d{2}-\d{2}$", args.date):
    parser.error(f"--date must be of the format 'YYYY-MM-DD', {args.date}")

info = {
        "date": np.datetime64(args.date, "D"),
        "latMin": min(args.latMin, args.latMax),
        "latMax": max(args.latMin, args.latMax),
        "lonMin": min(args.lonMin, args.lonMax),
        "lonMax": max(args.lonMin, args.lonMax),
        }

items = {
        "Cyclone": (args.cyclone, args.fnCyclone, 1),
        "Anticyclone": (args.anticyclone, args.fnAnticyclone, -1),
        }

for layer in items:
    (fn, shp, norm) = items[layer]
    with xr.open_dataset(fn) as ds:
        stime = time.time()
        gdf = mkContours(ds, info, norm, layer)
        print("Took", time.time()-stime, "to build", layer, "data")
        stime = time.time()
        gdf.to_file(shp)
        print("Took", time.time()-stime, "to write", shp)
