#! /usr/bin/env python3
#
# Create a list of target images from sdate to edate
#
# Dec-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import datetime

parser = ArgumentParser()
parser.add_argument("sdate", type=str, help="Starting date YYYY-MM-DD")
parser.add_argument("edate", type=str, help="Ending date YYYY-MM-DD")
parser.add_argument("--suffix", type=str, default=".png", help="File suffix")
parser.add_argument("--prefix", type=str, default="images/", help="File prefix")
parser.add_argument("--stepsize", type=int, default=10, help="File prefix")
args = parser.parse_args()

sdate = datetime.datetime.strptime(args.sdate, "%Y-%m-%d").date()
edate = datetime.datetime.strptime(args.edate, "%Y-%m-%d").date()
dt = datetime.timedelta(days=args.stepsize)
oneday = datetime.timedelta(days=1)

date = sdate

while date <= edate:
    date += dt
    print(args.prefix + str(min(edate, date-oneday)) + args.suffix)
