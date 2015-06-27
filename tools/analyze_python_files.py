#!/usr/bin/env python

import argparse
import os
import sys
import subprocess


def do_work(directory, output_base):
    if output_base[-1] != "/":
        output_base += "/"
    for dir_path, names, files in os.walk(directory):
        for i in files:
            if i.endswith(".py"):
                pyfile = dir_path + '/' + i
                out_name = pyfile.replace("/", "_")
                if out_name[0] == "_":
                    out_name = out_name[1:]
                out_name = output_base + out_name
                # run threshold identification provide output directory and no execution
                command = ['/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/thresholdanalysis/threshold_identification.py',
                           pyfile, '-i', out_name, '-n']
                subprocess.call(command)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description="Parse all of the python files in the directories below the one passed")
    parser.add_argument('directory', help="Directory to process")
    parser.add_argument('-o', '--output_directory', help="Directory to dump output files into")
    args = parser.parse_args()
    print args

    directory = args.directory
    if not os.path.exists(directory):
        print("Invalid directory!")
        sys.exit(-1)
    if args.output_directory:
        output_base = args.output_directory
        if not os.path.exists(output_base):
            output_base = directory
    else:
        output_base = directory
    directory = os.path.abspath(directory)
    output_base = os.path.abspath(output_base)
    do_work(directory, output_base)

