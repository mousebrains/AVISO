#! /usr/bin/env python3
#
# Fetch AVISO+ Mesoscale Eddy Product data via ftp
#
# Dec-2021, Pat Welch, pat@mousebrains.com

import fetch
import argparse

parser = argparse.ArgumentParser()
fetch.addArgs(parser)
grp = parser.add_argument_group(description="Directory related options")
grp.add_argument("--prefix", type=str, default="value-added/eddy-trajectory",
        help="Directory prefix to change to")
grp.add_argument("--version", type=str, default="3.1exp_DT_", help="Version to get data from")

gg = grp.add_mutually_exclusive_group()
gg.add_argument("--realtime", action="store_true", help="Use near-real-time directory")
gg.add_argument("--delayed", action="store_true", help="Use delayedd--time directory")

gg = grp.add_mutually_exclusive_group()
gg.add_argument("--allsat", action="store_true", help="Use all satellite directory")
gg.add_argument("--twosat", action="store_true", help="Use two satellite directory")
args = parser.parse_args()

directory = [args.prefix]
if args.delayed:
    nSat = "twosat" if args.twosat else "allsat" # Two satellites or all
    directory.append("delayed-time")
    directory.append("META" + args.version + nSat)
else:
    directory.append("near-real-time")

directory = "/".join(directory)

fetch.Fetch(args, directory)
