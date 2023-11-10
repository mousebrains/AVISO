#! /usr/bin/env python3
#
# Fetch, if needed, the AVISO+ Mesoscale Eddy Product
# generate plots in the specified date range, if needed
# Glue images into a movie
#
# Feb-2020, Pat Welch, pat@mousebrains.com

from TPWUtils import Logger
from TPWUtils.GreatCircle import Dist2Lon, Dist2Lat, Units
import logging
from argparse import ArgumentParser
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import geopandas as gpd
from shapely.geometry import Polygon, LinearRing, LineString
from ftplib import FTP
import matplotlib.pyplot as plt
import subprocess # To run ffmpeg for making the movie
import tempfile
import json
import re
import datetime
import os
import time
import sys

class RetrieveFile:
    def __init__(self, fn:str, targetSize:int, offset:int, qProgress:bool) -> None:
        self.__filename = fn
        self.__targetSize = targetSize / 1024 / 1024
        self.__qProgress = qProgress
        self.__fp = None
        self.__size = 0 if offset is None else offset
        self.__frac = None

    def __del__(self):
        if self.__fp is not None:
            self.__fp.close()
            self.__fp = None

    def block(self, data:bytes) -> None:
        if self.__fp is None:
            if self.__size: # Appending
                self.__fp = open(self.__filename, "ab")
            else: # Not appending
                self.__fp = open(self.__filename, "wb")
        self.__fp.write(data)
        self.__size += len(data)
        sz = self.__size / 1024 / 1024
        frac = "{:.1f}".format(100 * sz / self.__targetSize)
        if not self.__qProgress and (frac != self.__frac):
            logging.debug(f"{sz:.1f}/{self.__targetSize:.1f}MB {frac}%")
            self.__frac = frac

class FTPfetch:
    def __init__(self, args:ArgumentParser) -> None:
        self.__args = args
        self.__files = set()
        if args.nofetch: # Don't fetch new files, use existing ones
            self.__noFetch()
        else:
            self.__Fetch() # Fetch the files if needed
        if not args.ftpKeep: # Cleanup old files
            self.__cleanupFiles()

    @staticmethod
    def addArgs(parser:ArgumentParser) -> None:
        grp = parser.add_argument_group(description="FTP fetch related options")
        grp.add_argument("--nofetch", action="store_true", help="Don't fetch new files")
        grp.add_argument("--noprogress", action="store_true",
                help="Don't display download progress")
        grp.add_argument("--ftpDirectory", type=str,
                default="value-added/eddy-trajectory/near-real-time",
                help="Directory prefix to change to")
        grp.add_argument("--ftpSaveTo", type=str, default="data",
                help="Directory to save FTP files to")
        grp.add_argument("--ftpKeep", action="store_true", help="Don't clean up old data files")
        grp = parser.add_argument_group(description="Credentials related options")
        grp.add_argument("--ftpHost", type=str, metavar="foo.bar.com",
                default="ftp-access.aviso.altimetry.fr",
                help="Fully qualified hostname to connect to")
        grp.add_argument("--ftpCredentials", type=str, default="~/.config/AVISO/.aviso.credentials",
                help="Name of JSON file containinng the AVISO credentials")

    def cyclonic(self) -> str:
        for fn in self.__files:
            if re.search(r"_cyclonic_", fn):
                return fn
        raise Exception("No cyclonic files found")

    def anticyclonic(self) -> str:
        for fn in self.__files:
            if re.search(r"_anticyclonic_", fn):
                return fn
        raise Exception("No anticyclonic files found")

    def __getCredentials(self) -> tuple[str, str]:
        fn = self.__args.ftpCredentials
        try:
            with open(fn, "r") as fp:
                info = json.load(fp)
                if "username" in info and "password" in info:
                    return (info["username"], info["password"])
                logging.error("%s is not properly formated", fn)
        except Exception as e:
            logging.warning("Unable to open %s, %s", fn, str(e))

        logging.info("Going to build a fresh AVISO credentials file, %s", fn)
        info = {
            "username": input("Enter username:"),
            "password": input("Enter password:"),
            }

        os.makedirs(os.path.dirname(fn), mode=0o700, exist_ok=True)

        with open(fn, "w") as fp:
            json.dump(info, fp, indent=4, sort_keys=True)
        return (info["username"], info["password"])

    def __collectFiles(self) -> tuple[set, set]:
        directory = self.__args.ftpSaveTo
        old = set()
        items = {}
        for fn in os.listdir(directory):
            matches = re.match("Eddy_trajectory.*_(anticyclonic|cyclonic)_(\d+)_(\d+).nc", fn)
            if not matches: continue
            name = matches[1]
            info = {"filename": os.path.join(directory, fn),
                    "sdate": matches[2], "edate": matches[3]}
            if name not in items:
                items[name] = info
            elif info["edate"] < items[name]["edate"]: # info is older
                old.add(info["filename"])
            elif info["edate"] > items[name]["edate"]: # info is most current
                old.add(items[name]["filename"])
                items[name] = info
            elif info["sdate"] < items[name]["sdate"]: # info starts earlier
                old.add(items[name]["filename"])
                items[name] = info
            else: # info starts later
                old.add(info["filename"])

        current = set()
        for name in items: current.add(items[name]["filename"])

        return (current, old)

    def __cleanupFiles(self) -> None:
        (current, toDelete) = self.__collectFiles()
        for fn in toDelete:
            logging.info("Deleting %s", fn)
            os.unlink(fn)

    def __noFetch(self) -> None:
        (self.__files, toDelete) = self.__collectFiles()

    def __Fetch(self) -> None:
        (username, password) = self.__getCredentials()
        args = self.__args

        with FTP(host=args.ftpHost, user=username, passwd=password) as ftp:
            directory = args.ftpDirectory
            logging.info("CWD to %s", directory)
            ftp.cwd(directory)
            items = {}
            for item in ftp.mlsd(): # Get files to be downloaded, along with their information
                (fn, info) = item
                logging.debug("fn %s", fn)
                logging.debug("info %s", info)
                if "type" in info and info["type"] == "file":
                    items[fn] = info
                else:
                    logging.info("Skipping %s", fn)

            for fn in sorted(items): # Fetch the files, if needed
                item = items[fn]
                offset = None
                sz = int(item["size"])
                fnOut = os.path.join(args.ftpSaveTo, fn)
                self.__files.add(fnOut)
                if os.path.exists(fnOut):
                    info = os.stat(fnOut)
                    if info.st_size == sz:
                        logging.info("No need to fetch %s", fn)
                        continue
                    offset = info.st_size
                    logging.info(
                            "%s exists, {:.1f}MB left to fetch".format((sz - offset)/1024/1024),
                            fn)
                else:
                    logging.info("%s fetching {:.1f}MB".format(sz/1024/1024), fn)
 
                obj = RetrieveFile(fnOut, sz, offset, not args.noprogress)
                ftp.retrbinary(f"RETR {fn}", obj.block, blocksize=65536, rest=offset)

def mkCircles(radi:tuple[float], dist2lon:Dist2Lon, dist2lat:Dist2Lat,
              n:int=1000) -> gpd.GeoDataFrame:
    if radi is None or dist2lon is None: return None

    theta = np.linspace(-np.pi, np.pi, n)
    loci = np.array([np.cos(theta), np.sin(theta)]).T
    crs = "EPSG:4326" # Coordinate Reference System
    df = gpd.GeoDataFrame()
    for r in radi:
        rLon = r / dist2lon.distPerDeg # radius in degrees longitude
        rLat = r / dist2lat.distPerDeg # radius in degrees latitude
        lonLat = loci * np.array([rLon, rLat]) + [dist2lon.reference(), dist2lat.reference()]
        df = df.append(gpd.GeoDataFrame(data={
            "radius": (r,),
            "geometry": (LinearRing(lonLat),),
            },
            crs=crs))
    return df

def pruneData(ds:xr.Dataset, sdate:np.datetime64, edate:np.datetime64,
        latMin:float, latMax:float, lonMin:float, lonMax:float) -> xr.Dataset:
    # Prune in time
    q = np.logical_and(
            ds.time >= (sdate - np.timedelta64(30, "D")),
            ds.time <= edate)
    ds = ds.sel(obs=ds.obs[q])

    # Prune in latitude [-90,90]
    q = np.logical_and(ds.latitude >= latMin-1, ds.latitude <= latMax+1)
    ds = ds.sel(obs=ds.obs[q])

    # Handle the eddy walking across the prime merdian
    logging.info("Pre  Lon limits %s to %s", np.min(ds.longitude.data), np.max(ds.longitude.data))
    ds.longitude.data = np.remainder(ds.longitude.data, 360) # Wrap to [0, 360)
    logging.info("Post Lon limits %s to %s", np.min(ds.longitude.data), np.max(ds.longitude.data))

    # This will fail for lonMin/Max that span the prime merdian
    lonMin = np.remainder(lonMin, 360) # Map to [0,360)
    lonMax = np.remainder(lonMax, 360) # Map to [0, 360)
    q = np.logical_and(ds.longitude >= lonMin, ds.longitude <= lonMax)
    return ds.sel(obs=ds.obs[q])

def getContour(row:xr.Dataset, ds:xr.Dataset) -> xr.DataArray:
    if row.observation_flag: # Row does not have an observation
        ds = ds.sel(obs=ds.obs[ds.observation_flag == 0]) # Rows with observations
        if ds.obs.size == 0: # No matches found
            return (None, None, None)
        row = ds.sel(obs=ds.obs[-1]) # Time sorted, so -1 is most recent
    return (row.effective_contour_latitude - row.latitude,
            row.effective_contour_longitude - row.longitude,
            row.uavg_profile)

def mkContours(ds:xr.Dataset, date:np.datetime64) -> gpd.GeoDataFrame:
    crs = "EPSG:4326" # Coordinate Reference System
    ds = ds.sel(obs=ds.obs[ds.time <= date]) # Nothing in the future
    today = ds.sel(obs=ds.obs[ds.time == date])
    if today.obs.size == 0: # No data for today:
        return (None, None)

    df = gpd.GeoDataFrame() # speed contours
    track = gpd.GeoDataFrame() # recent track
    for obs in today.obs: # Walk through today's rows
        row = today.sel(obs=obs) # Select out a single observation
        trk = int(row.track) # Track number
        historical = ds.sel(obs=ds.obs[ds.track == trk])
        if historical.obs.size > 1: # Enough points to make a track
            lonLat = np.array([historical.longitude[-20:], historical.latitude[-20:]]).T
            track = track.append(gpd.GeoDataFrame(data={
                "trk": (int(row.track,)),
                "geometry": (LineString(lonLat),),
                },
                crs=crs,
                ))
        (latContour, lonContour, uAvg) = getContour(row, historical)
        if latContour is None: continue # Nothing to be done
        norm = np.linspace(1, 0, uAvg.size, endpoint=False); # [1,0)
        latContour = np.outer(latContour, norm) + float(row.latitude)
        lonContour = np.outer(lonContour, norm) + float(row.longitude)
        # Wrap lonContour to [-180,180)
        lonContour = np.fmod(lonContour + 180, 360)
        lonContour[lonContour < 0] += 360 
        lonContour -= 180
        for i in range(uAvg.size): # Walk through contours
            df = df.append(gpd.GeoDataFrame(data={
                "trk": (int(row.track),),
                "uAvg": (float(uAvg[i]),),
                "geometry": (Polygon(np.array([lonContour[:,i], latContour[:,i]]).T),),
                },
                crs=crs,
                ))
    return (df, track)

def doDate(dsC:xr.Dataset, dsA:xr.Dataset, date:np.datetime64, args:ArgumentParser,
        dist2lat:Dist2Lat, dist2lon:Dist2Lon, circles:gpd.GeoDataFrame,
        eez:gpd.GeoDataFrame) -> str:
    # Check to see if I need to make an image/shapefile

    png = os.path.join(args.png, f"{date}.png")
    shpC = os.path.join(args.shp, f"{date}_cyclonic.shp")
    shpA = os.path.join(args.shp, f"{date}_anticyclonic.shp")
    if os.path.isfile(png) and os.path.isfile(shpC) and os.path.isfile(shpA):
        logging.info("Skipping build for %s png and shp already exist", date)
        return png

    (contourC, trkC) = mkContours(dsC, date) # GeoDataFrame of cyclonic contours
    if contourC is None: 
        logging.info("No cyclonic data for %s", date)
        return None
    (contourA, trkA) = mkContours(dsA, date) # GeoDataFrame of anticyclonic contours
    if contourA is None: 
        logging.info("No anticyclonic data for %s", date)
        return None

    contourC = gpd.GeoDataFrame(contourC) # Get past a fiona typing issue
    contourA = gpd.GeoDataFrame(contourA) # Get past a fiona typing issue
    trkC     = gpd.GeoDataFrame(trkC) # Get past a fiona typing issue
    trkA     = gpd.GeoDataFrame(trkA) # Get past a fiona typing issue

    logging.info("Saving %s", shpC)
    contourC.to_file(shpC)
    logging.info("Saving %s", shpA)
    contourA.to_file(shpA)

    if dist2lon is not None:
        aspectRatio = dist2lat.distPerDeg / dist2lon.distPerDeg
        dLatdLon = (args.latMax - args.latMin) / (args.lonMax - args.lonMin)
        figRatio = aspectRatio * dLatdLon
    else:
        figRatio = 1

    tit = [np.datetime_as_string(date),
            "Blue-cyclonic",
            "Red-anticyclonic",
            ]

    (fig, ax) = plt.subplots(1,1, figsize=10 * np.array([1, figRatio]))
    if eez is not None: 
        eez.plot(ax=ax, color="green")
        tit.append("Green-EEZ")
    if circles is not None:
        circles = gpd.GeoDataFrame(circles)
        circles.plot(ax=ax, color="grey")
        tit.append("Grey-Circle")
    ax.plot(np.array(args.lonRef), np.array(args.latRef), "8k")
    contourC.plot(ax=ax, column="uAvg", vmin=0, vmax=0.5, cmap="Blues")
    contourA.plot(ax=ax, column="uAvg", vmin=0, vmax=0.5, cmap="Reds")
    trkC.plot(ax=ax, color="blue")
    trkA.plot(ax=ax, color="red")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True)
    ax.set_xlim(args.lonMin, args.lonMax)
    ax.set_ylim(args.latMin, args.latMax)
    if dist2lon is not None:
        ax.set_aspect(aspectRatio)
        ax2 = ax.secondary_xaxis("top",   functions=(dist2lon.deg2dist, dist2lon.dist2deg))
        ax3 = ax.secondary_yaxis("right", functions=(dist2lat.deg2dist, dist2lat.dist2deg))
        ax2.set_xlabel(", ".join(tit))
        ax3.set_ylabel("NM")
    logging.info("Saving %s", png)
    plt.savefig(fname=png, dpi=args.dpi)
    plt.close(fig)
    return png

def mkMovie(images:set[str], args:ArgumentParser) -> None:
    if not len(images): return # Nothing to build a movie from
    dates = {}
    for img in images:
        (date, ext) = os.path.splitext(os.path.basename(img))
        dates[np.datetime64(date)] = os.path.abspath(img)
    sdate = min(dates)
    edate = max(dates)
    fn = os.path.join(args.mp4, f"{sdate}.{edate}.mp4")
    if os.path.isfile(fn): # Check if we need to remake the movie by looking at modification times
        logging.info("Checking %s", fn)
        info = os.stat(fn)
        mtime = info.st_mtime
        logging.info("mtime %s", mtime)
        qRebuild = False
        for img in images:
            info = os.stat(img)
            if info.st_mtime >= mtime:
                qRebuild = True
                break
        if not qRebuild:
            logging.info("No reason to rebuild %s", fn)
            return
    logging.info("Rebuilding %s", fn)
    with tempfile.TemporaryDirectory() as tdir:
        cnt = 0
        for t in sorted(dates):
            img = dates[t]
            ofn = os.path.abspath(os.path.join(tdir, f"{cnt:04d}.png"))
            logging.info("%s -> %s", img, ofn)
            os.symlink(img, ofn)
            cnt += 1
        logging.info("Building %s", fn)
        cmd = ["ffmpeg",
                "-framerate", str(args.fps), # Frame rate in Hz
                "-i", os.path.join(tdir, "%04d.png"), # Input files
                "-vcodec", "libx264",
                "-crf", "27", # Quality, lower is better
                "-pix_fmt", "yuv420p", # Pixel color format
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", # Make sure both dimensions are even
                "-y", # answer yes to all questions, i.e. overwrite output
                fn, # Output filename
                ]
        sp = subprocess.run(cmd, shell=False, check=False, capture_output=True)
        if sp.returncode:
            logging.error("Executing %s", " ".join(cmd))
        else:
            logging.info("%s", "  ".join(cmd))
        if sp.stdout:
            try:
                logging.info("%s", str(sp.stdout, "utf-8"))
            except:
                logging.info("%s", sp.stdout)
        if sp.stderr:
            try:
                logging.info("%s", str(sp.stderr, "utf-8"))
            except:
                logging.info("%s", sp.stderr)

if __name__ == "__main__":
    parser = ArgumentParser()
    Logger.addArgs(parser)
    FTPfetch.addArgs(parser)
    parser.add_argument("--fetchonly", action="store_true", help="Only do FTP fetch")
    parser.add_argument("--skipmovies", action="store_true", help="Don't make a movie")
    grp = parser.add_argument_group(description="Output related options")
    grp.add_argument("--png", type=str, default="images", help="Output PNG image directory")
    grp.add_argument("--mp4", type=str, default="movies", help="Output MP4 movie directory")
    grp.add_argument("--shp", type=str, default="shapefiles", help="Output shapefile directory")
    grp = parser.add_argument_group(description="Image/Movie related options")
    grp.add_argument("--dpi", type=int, default=300, help="Image DPI")
    grp.add_argument("--fps", type=int, default=2, help="Frames/second")
    grp = parser.add_argument_group(description="Date selection options")
    grp.add_argument("--edate", type=str, help="Ending date, YYYY-MM-DD")
    grp0 = grp.add_mutually_exclusive_group()
    grp0.add_argument("--ndays", type=int, default=30,
                      help="Number of days to generate a movie for")
    grp0.add_argument("--sdate", type=str, help="Starting date, YYYY-MM-DD")
    grp.add_argument("--latMin", type=float, default=5, help="Minimum latitude, decimal degrees")
    grp.add_argument("--latMax", type=float, default=22.5, help="Maximum latitude, decimal degrees")
    grp.add_argument("--lonMin", type=float, default=135, help="Minimum longitude, decimal degrees")
    grp.add_argument("--lonMax", type=float, default=155, help="Maximum longitude, decimal degrees")
    grp = parser.add_argument_group(description="Graphical extras")
    grp.add_argument("--latRef", type=float, action="append",
        help="Geometric reference latitude, decimal degrees")
    grp.add_argument("--lonRef", type=float, action="append",
            help="Geometric reference longitude, decimal degrees")
    grp.add_argument("--circle", type=float, action="append",
            help="Circles of this many nautical miles about lat/lonRef[0]")
    grp.add_argument("--eez", type=str, 
            help="Filename of exclusive economic zone boundery geopackage file")
    args = parser.parse_args()

    Logger.mkLogger(args, fmt="%(asctime)s %(levelname)s: %(message)s", logLevel="INFO")

    logging.info("Args %s", args)

    try:
        logging.info("lat/lon Ref %s %s", args.latRef, args.lonRef)
        if ((args.latRef is None and args.lonRef is not None)
            or (args.latRef is not None and args.lonRef is None)
            ) and (len(args.latRef) != len(args.lonRef)):
            parser.error("You must specify the same number of instances of --latRef and --lonRef")

        edate = np.datetime64(datetime.date.today() - datetime.timedelta(days=14)) \
                if args.edate is None else np.datetime64(args.edate)
        if args.sdate:
            sdate = np.datetime64(args.sdate)
        else:
            sdate = edate - np.timedelta64(args.ndays, "D")

        if args.latRef is None:
            dist2lon = None
            dist2lat = None
        else:
            dist2lon = Dist2Lon(args.latRef[0], args.lonRef[0], Units.NauticalMiles)
            dist2lat = Dist2Lat(args.latRef[0], args.lonRef[0], Units.NauticalMiles)

        circles = mkCircles(args.circle, dist2lon, dist2lat)
        eez = None if args.eez is None else gpd.read_file(args.eez)

        os.makedirs(args.png, mode=0o755, exist_ok=True)
        os.makedirs(args.mp4, mode=0o755, exist_ok=True)
        os.makedirs(args.shp, mode=0o755, exist_ok=True)
        os.makedirs(args.ftpSaveTo, mode=0o755, exist_ok=True)

        a = FTPfetch(args)
        if args.fetchonly:
            logging.info("Only fetching")
            sys.exit(0)
        images = set()
        with xr.open_dataset(a.cyclonic()) as dsC, xr.open_dataset(a.anticyclonic()) as dsA:
            dsC = pruneData(dsC, sdate, edate, args.latMin, args.latMax, args.lonMin, args.lonMax)
            dsA = pruneData(dsA, sdate, edate, args.latMin, args.latMax, args.lonMin, args.lonMax)
            for date in np.arange(sdate, edate+np.timedelta64(1,"D")):
                # Loop over dates making images
                # Generate the image for a date, if needed
                fn = doDate(dsC, dsA, date, args, dist2lat, dist2lon, circles, eez)
                if fn is not None:
                    images.add(fn)

        if not args.skipmovies and len(images): # Some images to turn into a movie
            mkMovie(images, args)
    except:
        logging.exception("Args %s", args)
