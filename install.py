#! /usr/bin/env python3
#
# Install a service for AVISO service
#
# Feb-2022, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
import subprocess
import os
import sys

parser = ArgumentParser()
parser.add_argument("--service", type=str, default="AVISO.service", help="Service file")
parser.add_argument("--timer", type=str, default="AVISO.timer", help="Timer file")
parser.add_argument("--serviceDirectory", type=str, default="/etc/systemd/system",
        help="Where to copy service and timer files to")
parser.add_argument("--systemctl", type=str, default="/usr/bin/systemctl",
        help="systemctl executable")
parser.add_argument("--cp", type=str, default="/usr/bin/cp",
        help="cp executable")
parser.add_argument("--sudo", type=str, default="/usr/bin/sudo",
        help="sudo executable")
args = parser.parse_args()

root = os.path.abspath(os.path.expanduser(args.serviceDirectory))

fn = os.path.join(root, os.path.basename(args.service))
print("Writing to", fn)
subprocess.run((args.sudo, args.cp, args.service, fn), shell=False, check=True)

fn = os.path.join(root, os.path.basename(args.timer))
print("Writing to", fn)
subprocess.run((args.sudo, args.cp, args.timer, fn), shell=False, check=True)

print("Forcing reload of daemon")
subprocess.run((args.sudo, args.systemctl, "daemon-reload"), shell=False, check=True)

print(f"Enabling {args.service}")
subprocess.run((args.sudo, args.systemctl, "enable", args.service), shell=False, check=True)

print(f"Enabling {args.timer}")
subprocess.run((args.sudo, args.systemctl, "enable", args.timer), shell=False, check=True)

print(f"Starting {args.timer}")
subprocess.run((args.sudo, args.systemctl, "start", args.timer), shell=False, check=True)

print(f"Status {args.service} and {args.timer}")
s = subprocess.run((args.sudo, args.systemctl, "--no-pager", "status", args.service, args.timer),
        shell=False, check=False)

print(f"List timer {args.timer}")
s = subprocess.run((args.sudo, args.systemctl, "--no-pager", "list-timers", args.timer),
        shell=False, check=True)
