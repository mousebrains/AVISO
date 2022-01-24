#! /usr/bin/env python3
#
# Call Return a list of filenames
#
# Jan-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import numpy as np
import os.path
import re

parser = ArgumentParser()
parser.add_argument("--sdate", type=str, required=True, help="Starting date, YYYY-MM-DD")
parser.add_argument("--edate", type=str, required=True, help="Ending date, YYYY-MM-DD")
parser.add_argument("--directory", type=str, help="Output directory")
args = parser.parse_args()

if not re.match("^\d{4}-\d{2}-\d{2}$", args.sdate):
    parser.error(f"--sdate must be of the format 'YYYY-MM-DD', {args.date}")

if not re.match("^\d{4}-\d{2}-\d{2}$", args.edate):
    parser.error(f"--edate must be of the format 'YYYY-MM-DD', {args.date}")

sdate = np.datetime64(args.sdate, "D")
edate = np.datetime64(args.edate, "D")
dates = np.arange(sdate, edate+np.timedelta64(1, "D"))

for date in dates:
    fn = "image." + np.datetime_as_string(date, "D") + ".png"
    if args.directory:
        fn = os.path.join(args.directory, fn)
    print(fn)
