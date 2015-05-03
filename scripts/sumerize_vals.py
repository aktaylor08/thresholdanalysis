import argparse
from collections import defaultdict
import os
import pandas as pd
import threshold_node
import matplotlib.pyplot as plt
import glob
import json
import analysis_utils

from sys import stdout


def get_info(directory):
    info = {}
    for i in glob.glob(directory + "/*.json"):
        with open(i) as f:
            vals = json.load(f)
            for v in vals:
                info[v] = vals[v]
    return info



def get_df(bag_f, info):
    print 'Loading file {:s}'.format(bag_f)
    node = threshold_node.ThresholdNode(False)
    node.import_bag_file(bag_f,)
    thresh_df = node.get_new_threshold_data()
    param_keys = analysis_utils.get_param_keys(info)
    params_only = thresh_df[thresh_df['key'].apply(lambda param: param in param_keys)]
    return params_only

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
    for f in glob.glob(direct + '*.csv'):
        df = get_df(f, info)
        all_dfs[f] = df


    total_time = 0
    for f, df in all_dfs.iteritems():
        t = (df.index[-1] - df.index[0]).total_seconds()
        total_time += t
        print f, t
    print 'Total Runtime',  total_time






if __name__ == '__main__':
    main()
