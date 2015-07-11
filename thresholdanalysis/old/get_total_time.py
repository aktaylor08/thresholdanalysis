import argparse
import glob
import json

import numpy as np
import pandas as pd

import os

from thresholdanalysis.runtime import threshold_node
from thresholdanalysis.analysis.analysis_utils import get_info, get_df


def main():
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument("directory", help="Directory to parse")
    parser.add_argument('info_directory', help="Info Directory to parse")
    parser.add_argument('--namespace', )
    parser.add_argument('--namespace_thresh', )
    args = parser.parse_args()

    direct = args.directory
    if direct[-1] != '/':
        direct += '/'
    info = get_info(args.info_directory)
    all_dfs = {}
    if not os.path.exists(direct):
        print 'Path for bags does not exist!'
    else:
        print os.listdir(direct)

    for f in glob.glob(direct + '*.csv'):
        print f
        df = get_df(f, info)
        all_dfs[f] = df
        print 'hi',  f


    total_time = 0
    for f, df in all_dfs.iteritems():
        t = (df.index[-1] - df.index[0]).total_seconds()
        total_time += t
        print f, t
    print 'Total Runtime',  total_time

if __name__ == '__main__':
    main()
