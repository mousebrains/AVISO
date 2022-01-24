#! /usr/bin/env python3
#
# Fetch AVISO+ data via ftp
#
# Dec-2021, Pat Welch, pat@mousebrains.com

from argparse import ArgumentParser
from ftplib import FTP
import json
import os

def getCredentials(fn:str) -> tuple[str, str]:
    try:
        with open(fn, "r") as fp:
            info = json.load(fp)
            if "username" in info and "password" in info:
                return (info["username"], info["password"])
            print(fn, "is not properly formated")
    except Exception as e:
        print("Unable to open", fn, str(e))
    print("Going to build a fresh AVISO credentials file,", fn)
    info = {
            "username": input("Enter usrname:"),
            "password": input("Enter password:"),
            }
    with open(fn, "w") as fp:
        json.dump(info, fp, indent=4, sort_keys=True)
    return (info["username"], info["password"])

class RetrieveFile:
    def __init__(self, fn:str, targetSize:int, offset:int) -> None:
        self.__filename = fn
        self.__targetSize = targetSize / 1024 / 1024
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
        if frac != self.__frac:
            print("{:.1f}/{:.1f}MB {}%".format(sz, self.__targetSize, frac))
            self.__frac = frac

def addArgs(parser:ArgumentParser) -> None:
    grp = parser.add_argument_group(description="Credentials related options")
    grp.add_argument("--host", type=str, metavar="foo.bar.com",
            default="ftp-access.aviso.altimetry.fr",
            help="Fully qualified hostname to connect to")
    grp.add_argument("--credentials", type=str, default=".aviso.credentials",
            help="Name of JSON file containinng the AVISO credentials")

    grp = parser.add_argument_group(description="Output related options")
    grp.add_argument("--output", type=str, default=".", help="Output directory")

def Fetch(args:ArgumentParser, directory:str):
    (username, password) = getCredentials(args.credentials)

    with FTP(host=args.host, user=username, passwd=password) as ftp:
        print("CWD to", directory)
        ftp.cwd(directory)
        items = {}
        for item in ftp.mlsd(): # Get files to be downloaded, along with their information
            (fn, info) = item
            if "type" in info and info["type"] == "file":
                items[fn] = info
            else:
                print("Skipping", fn)

        for fn in sorted(items): # Fetch the files, if needed
            item = items[fn]
            offset = None
            sz = int(item["size"])
            if os.path.exists(fn):
                info = os.stat(fn)
                if info.st_size == sz:
                    print("No need to fetch", fn)
                    continue
                offset = info.st_size
                print(fn, "exists, {:.1f}MB left to fetch".format((sz - offset)/1024/1024))
            else:
                print(fn, "fetching {:.1f}MB".format(sz/1024/1024))

            obj = RetrieveFile(fn, sz, offset)
            ftp.retrbinary(f"RETR {fn}", obj.block, blocksize=65536, rest=offset)

if __name__ == "__main__":
    parser = ArgumentParser()
    addArgs(parser)
    parser.add_argument("--directory", type=str, required=True,
            help="Full path to fetch files from")
    args = parser.parse_args()

    Fetch(args, args.directory)
