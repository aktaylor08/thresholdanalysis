#!/usr/bin/env python
# encoding: utf-8
import argparse
from collections import defaultdict

import os
import rosbag_pandas as rbp
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

from IPython import embed


def add_to_store(store, key, idx, value, length):
    try:
        value = float(value)
    except ValueError:
        # Not a float
        pass
    if key in store:
        store[key][idx] = value
    else:
        if isinstance(value, int) or isinstance(value, float):
            arr = np.empty(length)
            arr.fill(np.NAN)
        else:
            arr = np.array([None] * length)
        arr[idx] = value
        store[key] = arr


def to_dataframe(series):
    errors = dict()
    length = len(series)
    datastore = {}
    index = np.array([None] * length)
    idx = 0
    for bag_time, i in series.iteritems():
        duplicate = False
        vals = i.split(',')
        time = pd.to_datetime(float(vals[0]), unit='s')

        loc = np.where(index == time)
        if len(loc[0]) > 0:
            print 'duplicate index', loc[0][0]
            new_idx = loc[0][0]
            old_idx = idx
            idx = new_idx
            duplicate = True

        index[idx] = time

        fname = vals[1]
        lineno = vals[2]
        thresh_id = fname + str(lineno)
        thresh_key = fname + ':' + lineno
        add_to_store(datastore, 'key', idx, thresh_key, length)
        add_to_store(datastore, 'id', idx, thresh_id, length)
        add_to_store(datastore, 'file_name', idx, fname, length)
        add_to_store(datastore, 'line_number', idx, lineno, length)

        line = vals[3]
        add_to_store(datastore, 'line', idx, line, length)
        truth = vals[4]
        add_to_store(datastore, 'result', idx, truth, length)
        thresholds = vals[5]
        add_to_store(datastore, 'thresholds', idx, thresholds, length)
        rest = vals[6:]
        for i in rest:
            try:
                key, val = i.split(':')
                add_to_store(datastore, key, idx, val, length)
            except ValueError as e:
                errors['Value Error: ' + thresh_key] = errors.get('Value Error: ' + thresh_key, 0) + 1

        if duplicate:
            idx = old_idx
        else:
            idx += 1
    index = index[:idx]
    for i, v in datastore.iteritems():
        datastore[i] = v[:idx]
    df = pd.DataFrame(data=datastore, index=index)
    print 'Errors: '
    for key, count in errors.iteritems():
        print count, key
    return df


def from_file(fname, return_both=False):
    df = rbp.bag_to_dataframe(fname)
    thresh = to_dataframe(df.threshold_information__data.dropna())
    if return_both:
        return thresh, df
    else:
        return thresh


def get_time_window(df, et, length):
    st = et - datetime.timedelta(0, 2.0)
    return df.between_time(st, et)


def get_goods_bads(df):
    '''get the good and bad markers'''
    try:
        bads = df.mark_bad__data_nsecs.dropna().index
    except Exception as e:
        print e
        bads = []
    try:
        goods = df.mark_good__data_nsecs.dropna().index
    except Exception as e:
        print e
        goods = []

    return bads, goods


def get_total_and_percentage_result(group):
    '''get the total and percentage for a number of groups'''
    vc = group['result'].value_counts()
    try:
        trues = vc['True']
    except:
        trues = 0

    try:
        falses = vc['False']
    except:
        falses = 0

    total = trues + falses
    percent = float(trues) / total
    return total, percent


def get_thresh_percents(df):
    '''get percentages and totals for all of the thresholds
    as a dictonary returned'''
    total = {}
    percent = {}
    all_groups = df.groupby('id')
    for name, group in all_groups:
        total[name], percent[name] = get_total_and_percentage_result(group)
    return total, percent


def get_bads_time(df):
    to_ret = []
    try:
        idx = df.mark_bad__data_nsecs.dropna().index
        vals = df.loc[idx, ['mark_bad__data_secs',  'mark_bad__data_nsecs']]
        for _, data in vals.iterrows():
            s = data['mark_bad__data_secs']
            ns = data['mark_bad__data_nsecs']
            time = s + ns / 1000000000.0
            to_ret.append(pd.to_datetime(time, unit='s'))
    except:
        print 'No bad marked?'
    return to_ret


def check_bad_vs_good(df, thresh):
    bads, goods = get_goods_bads(df)
    # bads =get_bads_time(df)
    fstring = '{:s}\t\t{:.2%}\t{:f}'
    total, percent = get_thresh_percents(thresh)
    for name in thresh['id'].unique():
        print fstring.format(name, percent[name],  total[name])
    print '\n\n'

    for et in bads:
        print et
        checks = get_time_window(thresh, et, 1)
        t, p = get_thresh_percents(checks)
        for name in t.keys():
            print '\t', fstring.format(name, p[name], t[name])


def ts_to_sec(ts):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = ts - epoch
    try:
        return delta.total_seconds()
    except:
        return np.NaN


def get_flops(thresh, cutoff):
    lows = []
    highs = []
    totals, percents = get_thresh_percents(thresh)
    for i in percents:
        if percents[i] < cutoff:
            lows.append(i)
        elif percents[i] > 1.0 - cutoff:
            highs.append(i)

    flops = {}
    # store the flops
    for tid, group in thresh.groupby('id'):
        if tid in lows:
            opposite = group[group.result == 'True']
            flops[tid] = opposite.index
        elif tid in highs:
            opposite = group[group.result == 'False']
            flops[tid] = opposite.index
    return flops


def add_times(thresh, flops):
    cur_time = thresh.index.to_series().apply(ts_to_sec)
    for name in flops.keys():
        thresh[name] = pd.Series(flops[name], flops[name])
        thresh[name] = thresh[name].apply(func=lambda x: pd.Timestamp(x))
        thresh[name] = thresh[name].apply(ts_to_sec)
        thresh[name] = cur_time - thresh[name].ffill()
    return thresh


def last_flop(df, thresh, cutoff=.25):
    print len(df)
    print len(thresh)
    # bads, goods = get_goods_bads(df)
    bads = get_bads_time(df)
    flops = get_flops(thresh, cutoff)
    thresh = add_times(thresh, flops)

    cols = [x for x in thresh.columns if x.startswith('/')]
    for time in bads:
        t = thresh.index.asof(time)
        data = thresh.loc[t, cols]
        data.sort()
        for k, v in data.iteritems():
            print k, v

    # embed()


def find_close(df, thresh, cutoff=.25):
    bads = get_bads_time(df)
    flops = get_flops(thresh, cutoff)
    thresh = add_times(thresh, flops)

    thresh['flop'] = None
    for name in flops:
        for val in flops[name]:
            thresh.loc[val, 'flop'] = name

    val_cols = [x for x in thresh.columns if x.find('value->') > 0]
    for v in val_cols:
        try:
            print v, len(thresh[v].value_counts())
        except:
            print '\n\n\n\n'
            print 'error ', v
            print '\n\n\n\n'
    # grouped = thresh.groupby('flop')
    # means =pd.DataFrame()
    # stds = pd.DataFrame()
    # for g,d in grouped:
    #     for col in val_cols:
    #         try:
    #             mean = d[col].mean()
    #             std = d[col].std()
    #             if pd.notnull(mean) and std != 0 and pd.notnull(std):
    #                 print g
    #                 print col
    #                 print mean
    #                 print std
    #                 print ''
    #                 means.loc[g, col] = mean
    #                 stds.loc[g, col] = std
    #         except Exception as e:
    #            # print d[col]
    #             print e

    # filled = thresh.ffill()
    # idx = []
    # for i in df.mark_bad__data_secs.dropna().index:
    #     idx.append(filled.index.asof(i))
    # points =  filled.loc[idx, val_cols]

    # for mark, row in points.iterrows():
    #     print mark
    #     for pos_thresh, vals in means.iterrows():
    #         print '\t', pos_thresh, np.abs((row - vals).dropna()).sum()

    # flop_df = thresh[thresh.flop.notnull()]

    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Extracting Data', usage='Use to extract threshold information',)
    parser.add_argument('-b', '--bag', help='The bag to extract information from', required=True)
    parser.add_argument('-e', '--extract_thresholds', help='Extract and save the thresholds to disk', action='store_true')
    parser.add_argument('-o', '--output', help='prefix to add to output files')
    parser.add_argument('-n', '--namespace', help='Namespace of the threshold topic')
    args = parser.parse_args()

    if args.namespace is not None:
        ns = args.namespace
    else:
        ns = ''
    fname,_ = os.path.splitext(args.bag)
    if args.output is not None:
        fname = args.output
    if args.extract_thresholds:
        df = rbp.bag_to_dataframe(args.bag, include=['/a/threshold_information'])

        if ns == '':
            data = df['threshold_information__data']
        else:
            data = df[ns + '_threshold_information__data']

        thresh = to_dataframe(data.dropna())
        thresh['line'] = thresh['line'].apply(lambda x: x.replace(',', ' '))


        thresh.to_csv( fname + '_thresh.csv', )

    # check_bad_vs_good(df,thresh)
    # last_flop(df, thresh)
    #find_close(df, thresh)
