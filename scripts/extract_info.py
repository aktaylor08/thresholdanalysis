#!/usr/bin/env python
# encoding: utf-8

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
    except:
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
    length = len(series)
    datastore = {}
    index = np.array([None] * length)
    idx = 0
    for bag_time, i in series.iteritems():
        vals = i.split(',')
        time = vals[0]
        index[idx] = pd.to_datetime(float(time), unit='s')

        fname = vals[1]
        lineno = vals[2]
        thresh_id = fname + str(lineno) 
        add_to_store(datastore, 'id', idx, thresh_id, length)
        add_to_store(datastore, 'file_name', idx, fname, length)
        add_to_store(datastore, 'line_number', idx, lineno, length)

        line = vals[3]
        add_to_store(datastore, 'line', idx, line, length)
        truth = vals[4]
        add_to_store(datastore, 'result', idx, truth, length)
        rest = vals[5:]
        for i in rest:
            try:
                key, val = i.split(':')
            except:
                pass
                # print i 
                # vals = i.split(':')
                # key = vals[0]
                # print key
                # val = ''.join(vals[1:])


            add_to_store(datastore, key, idx, val, length)
        idx +=1
    df = pd.DataFrame(data=datastore, index=index)
    return df

def from_file(fname, return_both=False):
    df = rbp.bag_to_dataframe(fname)
    thresh = to_dataframe(df.threshold_information__data.dropna())
    if return_both:
        return thresh, df
    else:
        return thresh


def get_time_window(df, et, length):
        st = et - datetime.timedelta(0,2.0)
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
    idx = df.mark_bad__data_nsecs.dropna().index
    vals = df.loc[idx,['mark_bad__data_secs',  'mark_bad__data_nsecs']]
    for _, data in vals.iterrows():
        s = data['mark_bad__data_secs']
        ns = data['mark_bad__data_nsecs']
        time = s + ns / 1000000000.0
        to_ret.append(pd.to_datetime(time, unit='s'))
    return to_ret

def check_bad_vs_good(df, thresh):
    bads, goods =  get_goods_bads(df)
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


def last_flop(df, thresh, cutoff=.1):
    print len(df)
    print len(thresh)
    # bads, goods = get_goods_bads(df)
    bads = get_bads_time(df)
    lows = []
    highs = []
    totals, percents = get_thresh_percents(thresh)
    for i in percents:
        if percents[i] < cutoff:
            lows.append(i)
        elif percents[i] > 1.0 - cutoff:
            highs.append(i)

         
    flops = {}
    #store the flops
    for tid, group in thresh.groupby('id'):
        if tid in lows:
            opposite = group[group.result == 'True']
            flops[tid] = opposite.index
        elif tid in highs:
            opposite = group[group.result == 'False']
            flops[tid] = opposite.index

    for name in flops.keys():
        thresh[name] = pd.Series(flops[name], flops[name])
        thresh[name] = (thresh.index - thresh[name].ffill()) / np.timedelta64(1, 's')


    cols = [x for x in thresh.columns if x.startswith('/')]
    for time in bads:
        t = thresh.index.asof(time)
        data = thresh.loc[t, cols]
        data.sort()
        for k,v in data.iteritems():
            print k, v

        

    embed()






if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'need file'
        sys.exit()

    f = sys.argv[1]
    df = rbp.bag_to_dataframe(f)
    data = df.threshold_information__data
    thresh = to_dataframe(data.dropna())
    check_bad_vs_good(df,thresh)
    last_flop(df, thresh)



