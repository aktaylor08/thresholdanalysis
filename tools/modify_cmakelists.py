#!/usr/bin/env python

import argparse
import os
import sys
import re

INSTRUMENT_FILE_LOCATION = "/home/ataylor/"


def get_groups(lines):
    arguments = []
    cur_cmd = None
    pcount = 0
    inside_args = []
    active = False
    for i in lines:
        print i
        match = re.search("(\w*\s*\()", i)
        if match is not None:
            pcount += 1
            if pcount == 1:
                cur_cmd = match.group()[:-1]
            else:
                print match.group()

        match = re.search("\w*\s*\)", i)
        if match is not None:
            pcount -= 1
            arguments.append((cur_cmd, inside_args))
    print arguments




def do_work(directory_start ):
    for dir_path, names, files in os.walk(directory_start):
        if "CMakeLists.txt" in files and "package.xml" in files:
            cmake = dir_path + '/CMakeLists.txt'
            to_parse = []
            with open(cmake) as cfile:
                lines = cfile.read().split("\n")
                # get rid of comments
                for i in lines:
                    cidx = i.find("#")
                    if cidx < 0:
                        to_parse.append(i)
                    elif cidx == 0:
                        pass
                    else:
                        to_parse.append(i[:cidx])
                groups = get_groups(to_parse)




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
