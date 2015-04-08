__author__ = 'ataylor'

#!/usr/bin/env python

import argparse
import glob

def read_data(directory):
    for json_file in glob.glob(directory + '*.json'):
        print json_file


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description=("Read in information and report information about"
                     "thresholds found during compilation in a ros package"))
    parser.add_argument('directory', help="Directory to process")
    args = parser.parse_args()
    directory = args.directory
    if directory[-1] != '/':
        directory += '/'
    data = read_data(directory)
