#!/usr/bin/env python

import argparse
import os
import sys

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="Parse all of the python files in the directories below the one passed")
    parser.add_argument('directory', help="Directory to process")
    parser.add_argument('-o', '--output_directory',  help="Directory to dump output files into")
    args = parser.parse_args()

    directory = parser.directory
    if not os.path.exists(directory):
        print("Invalid directory!")
        sys.exit(-1)

