#!/usr/bin/env python
# encoding: utf-8
import argparse

import os
import math
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


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


def get_thresh_info(thresh_df):
    keys = []
    trues = []
    falses = []
    totals = []
    pts = []
    pfs = []

    #get flop information as well as number of true and false ect.
    for k, v in thresh_df.groupby('key'):
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
    # create the dataframe
    # now add flop_times
    thresh_df['last_flop'] = np.NaN
    for i in df.index:
        flop_locations = get_flops(thresh_df, i)
        thresh_df = add_times(thresh_df, i, flop_locations)
    return df, thresh_df


def add_times(to_add_df, key, locations):
    vals = to_add_df[to_add_df['key'] == key]
    cur_time = vals.index.to_series().apply(ts_to_sec)
    data = pd.Series(index=vals.index)
    for idx in locations:
        data[idx] = idx
    data = data.apply(func=lambda x: pd.Timestamp(x))
    data = data.apply(ts_to_sec)
    data = data.ffill()
    for idx, val in data.iteritems():
        to_add_df.loc[idx, 'last_flop'] = cur_time[idx] - val
    return to_add_df


def ts_to_sec(ts):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = ts - epoch
    try:
        return delta.total_seconds()
    except:
        return np.NaN


def get_times(fname):
    if not os.path.isfile(fname):
        print 'The file does not exist'
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


def handle_no_advance(thresh, flops, no_adv):
    pass


def handle_advance(thresh, flops, adv, time_limit=8.0):
    groups = thresh.groupby('key')
    for marked_time in adv:
        et = marked_time + datetime.timedelta(seconds=time_limit)
        st = marked_time - datetime.timedelta(seconds=time_limit)
        last_flops = {}
        # get the time from the last flop
        for key, data in groups:
            idx = data.index.asof(marked_time)



            if not isinstance(idx, float):
                if idx > st:
                    time = data.loc[idx, 'last_flop']
                    if not math.isnan(time):
                        last_flops[key] = time

        # sort and filter
        x = [(k, v) for k, v in last_flops.iteritems()]
        x = filter(lambda arg: arg[1] < time_limit, x)
        x = sorted(x, key=lambda arg: arg[1])

        # report
        in_limits = thresh.between_time(st, et)
        print 'Advance:', marked_time
        if len(x) > 0:
            for flop in x:
                values = in_limits[in_limits['key'] == flop[0]]
                size = len([k for k in values.columns if k.startswith('res_') and len(values[k].dropna() > 1)])
                if size > 1:
                    fig, ax = plt.subplots(size)
                    for i in range(size):
                        k1 = 'cmp_{:d}_0'.format(i)
                        k2 = 'const_{:d}_0'.format(i)
                        ax[i].plot(values[k1])
                        ax[i].plot(values[k2])
                        a = ax[i].get_ylim()
                        mid = (a[0] + a[1]) / 2.0
                        # ax[i].scatter([marked_time], [mid], marker='*', c='g', s=40)
                    plt.show()

                else:
                    fig, ax = plt.subplots()
                    print values['cmp_0_0'].values.head(4)
                    ax.plot(values['cmp_0_0']) # .index, values['cmp_0_0'].values)
                    ax.plot(values['const_0_0']) #.index, values['const_0_0'].values)
                    a = ax.get_ylim()
                    mid = (a[0] + a[1]) / 2.0
                    s = pd.Series(data=[mid], index=[marked_time])
                    ax.scatter(s.index, s.values)
                    plt.show()

                print '\t', size
                print '\t', flop[1], flop[0], len(in_limits[in_limits['key'] == flop[0]])

        else:
            print 'No flops within {:f}'.format(time_limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument('-t', '--thresholds', required=True)
    parser.add_argument('-b', '--bag', required=True)
    args = parser.parse_args()

    thresh = get_thresholds(args.thresholds)
    flops, thresh = get_thresh_info(thresh)
    adv, no_adv = get_times(args.bag)
    handle_no_advance(thresh, flops, no_adv)
    handle_advance(thresh, flops, no_adv)
