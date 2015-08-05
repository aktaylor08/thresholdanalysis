from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from thresholdanalysis.analysis import analysis_utils
import glob
import csv

import argparse


def main():
    parser = argparse.ArgumentParser('Create general statistics here')
    parser.add_argument('thresh_dir',)
    parser.add_argument('info_dir',)
    parser.add_argument('csv_dir',)
    parser.add_argument('--output-prefix')
    args = parser.parse_args()
    direct = args.thresh_dir
    output = args.output_prefix
    csv_dir = args.csv_dir
    if output is not None:
        print "Outputting with prefix", output
    else:
        output = 'generic_'
    info_dir = args.info_dir
    if direct[-1] != '/':
        direct += '/'
    if info_dir[-1] != '/':
        info_dir += '/'
    info_dir = args.info_dir
    if csv_dir[-1] != '/':
        csv_dir += '/'
    info = analysis_utils.get_info(info_dir)
    all_dfs = {}
    for f in glob.glob(direct + '*.csv'):
        df = analysis_utils.get_df(f, info)
        all_dfs[f] = df
    print len(all_dfs)
    # add some information
    for f, df in all_dfs.iteritems():
        df['source'] = df['key'].apply(lambda x: info[x]['source'])
        df['file'] = df['key'].apply(lambda x: info[x]['file'])
        df['lineno'] = df['key'].apply(lambda x: info[x]['lineno'])

    print '\n'
    keys = sorted(all_dfs.keys())
    for i in keys:
        for k, v in all_dfs[i].groupby('source'):
            otras = defaultdict(list)
            other_files = defaultdict(list)
            things = None
            myval = 0
            for j in keys:
                for ok, ov in all_dfs[j].groupby('source'):
                    if ok == k:
                        if ov['thresh'][0] != v['thresh'][0]:
                            otras[k].append(ov['thresh'][0])
                            other_files[k].append(j)
                            myval = v['thresh'][0]
                            things = v['key'].unique()
                            # print i
                            # print '\t', k,  v['thresh'][0]
                            # print j
                            # print '\t', ok,  ov['thresh'][0]
                            # print '\t\t', v['key'].unique()
                            # print '=========================='
                            # print '\n'
                            # print '\n'
                            # break
            if len(otras) > 0:
                print i
                print myval
                print otras
                print other_files
                print things
                print '\n'

if __name__ == '__main__':
    main()
