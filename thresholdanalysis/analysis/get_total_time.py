import argparse
import glob
import json

import numpy as np
import pandas as pd

import os

from thresholdanalysis.runtime import threshold_node
from thresholdanalysis.analysis import analysis_utils


def get_info(directory):
    info = {}
    for i in glob.glob(directory + "/*.json"):
        with open(i) as f:
            vals = json.load(f)
            for v in vals:
                info[v] = vals[v]
    return info

def calculate_ranking(score_df, zero_bad=False):
    cols = score_df.columns
    num = len(cols)
    col_map = {v: k for k,v in enumerate(cols)}
    idxes = []
    store = np.zeros((len(score_df), len(cols)))
    print store.shape
    rc = 0
    for idx, row in score_df.iterrows():
        idxes.append(idx)
        row.sort()
        count = 0
        for name, val in row.iteritems():
            col = col_map[name]
            if zero_bad:
                if val < 9999:
                    store[rc, col] = 1 - ((float(count)) / num)
                    count += 1
            else:
                store[rc, col] = 1 - ((float(count)) / num)
                count += 1
        rc += 1
    return pd.DataFrame(data=store, index=idxes, columns=cols)






def get_df(bag_f, info, parsed=True):
    if not parsed:
        print 'Loading file {:s}'.format(bag_f)
        node = threshold_node.ThresholdNode(False)
        node.import_bag_file(bag_f,)
        thresh_df = node.get_new_threshold_data()
    else:
        thresh_df = pd.read_csv(bag_f, index_col=0, parse_dates=Tru)
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
