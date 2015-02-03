#!/usr/bin/env python
# encoding: utf-8
import argparse
from collections import defaultdict

import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def handle_no_advance(thresh, spots):
    pass


def handle_advance(thresh, spots):
    pass


def get_thresholds(fname):
    if os.path.isfile(fname):
        df = pd.read_csv(fname, parse_dates=True, index_col=0)
        return df
    else:
        print('Threshold file does not exist')
        sys.exit()


def get_flops(df, key):
    data_spot = df[df['key'] == key]
    value = data_spot['result'][0]
    flopped = False
    flop_times = []
    for time, res in data_spot['result'].iteritems():
        if res != value:
            if not flopped:
                flop_times.append(time)
                flopped = True
        else:
            flopped = False
    return flop_times


def get_thresh_info(thresh, limit=.6):
    keys = []
    trues = []
    falses = []
    totals = []
    pts = []
    pfs = []
    for k,v in thresh.groupby('key'):
        plt.show()
        counts = v['result'].value_counts()
        if True in counts:
            t = counts[True]
        else:
            t = 0
        if False in counts:
            f = counts[False]
        else:
            f = 0
        keys.append(k)
        trues.append(t)
        falses.append(f)
        totals.append(t+f)
        pt = (t / (float(t+f)))
        pf = (f / (float(t+f)))
        pts.append(pt)
        pfs.append(pf)

    df = pd.DataFrame(data={'true_count' : trues, 'false_count' : falses, 'count' : totals,
                            'true_prop': pts, 'false_prop' : pfs}, index=keys)
    thresh['last'] = np.NaN
    print thresh['last']
    for i in df.index:
        flops = get_flops(thresh, i)
        thresh = add_times(thresh, i, flops)
        print flops
    return df


def add_times(thresh, key, flops):
    vals = thresh[thresh['key'] == key]
    cur_time = vals.index.to_series().apply(ts_to_sec)
    vals['last'] = pd.Series(flops, flops)
    vals['last'] = vals['last'].apply(func=lambda x: pd.Timestamp(x))
    vals['last'] = vals['last'].apply(ts_to_sec)
    vals['last'] = cur_time - vals['last'].ffill()
    for idx, val in vals['last'].iteritems():
        thresh.ix['last', idx] = val
    return thresh

def ts_to_sec(ts):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = ts - epoch
    try:
        return delta.total_seconds()
    except:
        return np.NaN


def get_times(fname):
    if not os.path.isfile(fname):
        print 'The file does not exisit'
        sys.exit(-1)
    df = pd.read_csv(fname, parse_dates=True, index_col=0)
    no_action = []
    action = []
    if 'mark_no_action__data_nsecs' in df.columns:
        idx = df.mark_no_action__data_nsecs.dropna().index
        vals = df.loc[idx, ['mark_no_action__data_secs',  'mark_no_action__data_nsecs']]
        for _, data in vals.iterrows():
            s = data['mark_no_action__data_secs']
            ns = data['mark_no_action__data_nsecs']
            time = s + ns / 1000000000.0
            no_action.append(pd.to_datetime(time, unit='s'))
    else:
        print 'No bad marked?'
    if 'mark_action__data_nsecs' in df.columns:
        idx = df.mark_action__data_nsecs.dropna().index
        vals = df.loc[idx, ['mark_action__data_secs',  'mark_action__data_nsecs']]
        for _, data in vals.iterrows():
            s = data['mark_action__data_secs']
            ns = data['mark_action__data_nsecs']
            time = s + ns / 1000000000.0
            action.append(pd.to_datetime(time, unit='s'))
    return action, no_action

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument('-t', '--thresholds', required=True)
    parser.add_argument('-b', '--bag', required=True)
    args = parser.parse_args()


    thresh = get_thresholds(args.thresholds)
    flops = get_thresh_info(thresh)
    adv, no_adv = get_times(args.bag)

