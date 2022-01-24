#! /usr/bin/env python3
#
# Plot the AVISO Mesoscale Eddy product data for a day
#
# Jan-2022, Pat Welch

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import time
import os
# import logging
import re
import sys

def greatCircle(lon0:np.array, lat0:np.array, lon1:np.array, lat1:np.array, re=3443.92):
    ''' Radius of earth in nautical miles '''
    lon0 = np.radians(lon0)
    lat0 = np.radians(lat0)
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    return re * np.arccos( \
            np.sin(lat0) * np.sin(lat1) + \
            np.cos(lat0) * np.cos(lat1) * np.cos(lon0 - lon1) \
            )

class DistanceDegree:
    def __init__(self, distPerDeg:float, degRef:float) -> None:
        self.distPerDeg = distPerDeg
        self.__degRef = degRef

    def deg2dist(self, deg:np.array) -> np.array:
        return (deg - self.__degRef) * self.distPerDeg

    def dist2deg(self, dist:np.array) -> np.array:
        return self.__degRef + dist / self.distPerDeg

class Dist2Lon(DistanceDegree):
    def __init__(self, latRef:float, lonRef:float) -> None:
        DistanceDegree.__init__(self, greatCircle(lonRef-0.5, latRef, lonRef+0.5, latRef), lonRef)

class Dist2Lat(DistanceDegree):
    def __init__(self, latRef:float, lonRef:float) -> None:
        DistanceDegree.__init__(self, greatCircle(lonRef, latRef-0.5, lonRef, latRef+0.5), latRef)

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

def plotEddies(ax, ds:xr.Dataset, info:dict, args:ArgumentParser, cm, tit:str) -> None:
    stime = time.time()
    q = np.logical_and.reduce((
        ds.time == info["date"],
        ds.latitude  >= info["latMin"] - 2, # Extra for the contours at the edge
        ds.latitude  <= info["latMax"] + 2,
        ds.longitude >= wrapTo360(info["lonMin"]) - 2, # Extra for the contours at the edge
        ds.longitude <= wrapTo360(info["lonMax"]) + 2,
        ))
    df = pd.DataFrame()
    df["t"] = ds.time[q]
    df["trk"] = ds.track[q]
    df["indices"] = np.nonzero(q)[0]
    df["oFlag"] = ds.observation_flag[q]
    df["lat"] = ds.latitude[q]
    df["lon"] = ds.longitude[q]
    nAvg = ds.uavg_profile.shape[1]
    scaling = np.arange(nAvg, 0, -1) / nAvg # contour scaling
    cNorm = args.uFullScale # Color normalization
    for index, row in df.iterrows():
        (cLat, cLon, uAvg) = fetchContour(ds, row)
        lon = wrapTo180(row.lon) # For display purposes
        uAvg = uAvg / cNorm
        for i in range(nAvg):
            ax.fill(
                cLon * scaling[i] + lon,
                cLat * scaling[i] + row.lat,
                color=cm(uAvg[i]),
                alpha=0.5)
        if args.nBack: # plot the past trajectory
            indices = row.indices + np.arange(-args.nBack, 1) # Indices to look for historical information in
            trk = ds.track[indices].to_numpy()
            indices = indices[trk == row.trk] # Information only for this track
            lat = ds.latitude[indices]
            lon = wrapTo180(ds.longitude[indices])
            ax.plot(lon, lat, "-", color=cm(0.5))

parser = ArgumentParser()
grp = parser.add_argument_group(description="Input NetCDF file options")
grp.add_argument("--cyclone", type=str, required=True, help="Cyclone NetCDF data")
grp.add_argument("--anticyclone", type=str, required=True, help="Anticyclone NetCDF data")

grp = parser.add_argument_group(description="Plot related options")
grp.add_argument("--cmCyclones", type=str, default="Blues", help="Color map for cyclone")
grp.add_argument("--cmAnticyclones", type=str, default="Reds", help="Color map for anticyclone")
grp.add_argument("--uFullScale", type=float, default=0.5, help="uAvg full scale value")
grp.add_argument("--latRef", type=float, action="append",
        help="Geometric reference latitude, decimal degrees")
grp.add_argument("--lonRef", type=float, action="append",
        help="Geometric reference longitude, decimal degrees")
grp.add_argument("--nBack", type=int, default=20, help="Number of historical days to plot tracks for")

grp = parser.add_argument_group(description="Output plot file options")
grp.add_argument("--save", type=str, help="Output filename")
grp.add_argument("--dpi", type=int, default=300, help="Output DPI")

grp = parser.add_argument_group(description="Data selection criteria options")
grp.add_argument("--date", type=str, default="2019-03-01", help="Date to plot data for, YYYY-MM-DD")
grp.add_argument("--latMin", type=float, default=5, help="Minimum latitude, decimal degrees")
grp.add_argument("--latMax", type=float, default=30, help="Maximum latitude, decimal degrees")
grp.add_argument("--lonMin", type=float, default=130, help="Minimum longitude, decimal degrees")
grp.add_argument("--lonMax", type=float, default=155, help="Maximum longitude, decimal degrees")

args = parser.parse_args()

if not re.match("^\d{4}-\d{2}-\d{2}$", args.date):
    parser.error(f"--date must be of the format 'YYYY-MM-DD', {args.date}")

if args.latRef is None and args.lonRef is None:
    args.latRef = [ 13.4443,  23.6978,   7.5150] # Guam+Taiwan+Palau
    args.lonRef = [144.7937, 120.9605, 134.5825] # Guam+Taiwan+Palau

if (args.latRef is None and args.lonRef is not None) or \
        (args.latRef is not None and args.lonRef is None) or \
        (len(args.latRef) != len(args.lonRef)):
    parser.error("You must specify the same number of instances of --latRef and --lonRef")

info = {
        "date": np.datetime64(args.date, "D"),
        "latMin": min(args.latMin, args.latMax),
        "latMax": max(args.latMin, args.latMax),
        "lonMin": min(args.lonMin, args.lonMax),
        "lonMax": max(args.lonMin, args.lonMax),
        }

dist2lon = Dist2Lon(args.latRef[0], args.lonRef[0])
dist2lat = Dist2Lat(args.latRef[0], args.lonRef[0])

fig, ax = plt.subplots(figsize=(10,10)) # Figure to plot
ax.grid(True)
ax.plot(np.array(args.lonRef), np.array(args.latRef), "8k")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xlim(info["lonMin"], info["lonMax"])
ax.set_ylim(info["latMin"], info["latMax"])
ax.set_aspect(dist2lat.distPerDeg / dist2lon.distPerDeg)
ax2 = ax.secondary_xaxis("top",   functions=(dist2lon.deg2dist, dist2lon.dist2deg))
ax3 = ax.secondary_yaxis("right", functions=(dist2lat.deg2dist, dist2lat.dist2deg))
ax2.set_xlabel(args.date)
ax3.set_ylabel("NM")

with xr.open_dataset(args.cyclone) as ds:
    plotEddies(ax, ds, info, args, plt.get_cmap(args.cmCyclones), "Cyclone")

with xr.open_dataset(args.anticyclone) as ds:
    plotEddies(ax, ds, info, args, plt.get_cmap(args.cmAnticyclones), "Anticyclone")

if args.save:
    dirname = os.path.dirname(args.save)
    os.makedirs(dirname, mode=0o775, exist_ok=True)
    plt.savefig(
            fname=args.save, 
            dpi=args.dpi,
            metadata={
                "cyclone": args.cyclone,
                "anticyclone": args.anticyclone,
                },
            )
else: # Plot on screen
    plt.show()
