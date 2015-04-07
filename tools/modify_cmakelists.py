#!/usr/bin/env python

import argparse
import os
import sys
import subprocess


def do_work(directory ):
    for dir_path, names, files in os.walk(directory):
        for i in files:
            if i == "CMakeLists.txt":
                print i



if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description="Parse all of the python files in the directories below the one passed")
    parser.add_argument('directory', help="Directory and below to process")
    args = parser.parse_args()

    directory = args.directory
    if not os.path.exists(directory):
        print("Invalid directory!")
        sys.exit(-1)
    do_work(directory)
