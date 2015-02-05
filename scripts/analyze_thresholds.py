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


class GraphInfo(object):

    def __init__(self, index, series, names, scatter_data, suggestion):
        self.index = index
        self.series = series
        self.names = names
        self.scatter_data = scatter_data
        self.suggestion = suggestion


class ThreshStatement(object):

    def __init__(self, threshold, stmt_key, score, suggestion=None):
        self.threshold = threshold
        self.stmt_key = stmt_key
        self.score = score
        self.suggestion=suggestion

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{:f} {:s} {:s}'.format(self.score, self.threshold, self.stmt_key)


class ThreshInfoStore(object):

    def __init__(self):
        self._stmt_map = {}
        self._thresh_map = {}
        self.graph_map = {}
        self.thresh_list = []
        self.sorted = False

    def import_data(self, thresholds):
        for i in thresholds:
            self.import_thresh(i)

    def sort(self):
        self.thresh_list = sorted(self.thresh_list, key=lambda x: x.score)
        self.sorted = True

    def import_thresh(self, incoming):
        self.thresh_list.append(incoming)
        self.sorted = False

        # put it in this map
        if incoming.stmt_key in self._stmt_map:
            self._stmt_map[incoming.stmt_key].append(incoming)
        else:
            self._stmt_map[incoming.stmt_key] = [incoming]

        # put it in this map
        if incoming.threshold in self._thresh_map:
            self._thresh_map[incoming.threshold].append(incoming)
        else:
            self._thresh_map[incoming.threshold] = [incoming]

    def get_entries(self, key, threshold):
        vals = []
        for i in self.thresh_list:
            if i.stmt_key == key and i.threshold == threshold:
                vals.append(i)
        return vals

    def add_graph(self, key, info):
        self.graph_map[key] = info

    def drop_low_scores(self, req_score):
        self.thresh_list = filter(lambda x: x.score > req_score, self.thresh_list)


def get_thresholds(fname):
    if os.path.isfile(fname):
        df = pd.read_csv(fname, parse_dates=True, index_col=0)
        return df
    else:
        print('Threshold file does not exist')
        sys.exit()


def get_series_flops(series):
    flopped = False
    flop_times = []
    value = series[0]
    for time, res in series.iteritems():
        if res != value:
            if not flopped:
                flop_times.append(time)
                flopped = True
        else:
            flopped = False
    return flop_times


def get_flops(df, key):
    data_spot = df[df['key'] == key]
    series = data_spot['result']
    return get_series_flops(series)


def get_thresh_info(thresh_df):
    keys = []
    trues = []
    falses = []
    totals = []
    pts = []
    pfs = []

    # get flop information as well as number of true and false ect.
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
        totals.append(t + f)
        pt = (t / (float(t + f)))
        pf = (f / (float(t + f)))
        pts.append(pt)
        pfs.append(pf)

    df = pd.DataFrame(data={'true_count': trues, 'false_count': falses, 'count': totals,
                            'true_prop': pts, 'false_prop': pfs}, index=keys)
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
        vals = df.loc[idx, ['mark_no_action__data_secs', 'mark_no_action__data_nsecs']]
        for _, data in vals.iterrows():
            s = data['mark_no_action__data_secs']
            ns = data['mark_no_action__data_nsecs']
            time = s + ns / 1000000000.0
            no_action.append(pd.to_datetime(time, unit='s'))
    else:
        print 'No bad marked?'
    if 'mark_action__data_nsecs' in df.columns:
        idx = df.mark_action__data_nsecs.dropna().index
        vals = df.loc[idx, ['mark_action__data_secs', 'mark_action__data_nsecs']]
        for _, data in vals.iterrows():
            s = data['mark_action__data_secs']
            ns = data['mark_action__data_nsecs']
            time = s + ns / 1000000000.0
            action.append(pd.to_datetime(time, unit='s'))
    return action, no_action


def handle_no_advance(thresh_df, flop_info, no_advances, time_limit=15.0):
    groups = thresh_df.groupby('key')
    results = {}
    for marked_time in no_advances:
        scores = []
        thresh_store = ThreshInfoStore()
        et = marked_time + datetime.timedelta(seconds=time_limit)
        st = marked_time - datetime.timedelta(seconds=time_limit)
        for key, data in groups:
            idx = data.index.asof(marked_time)
            needed_info = []
            graph_list = []
            if not isinstance(idx, float):
                before = data.between_time(st, idx)
                total = data.between_time(st, et)
                cols = [x for x in total.columns if x.startswith('res_') and len(total[x].dropna()) > 0]
                if len(cols) == 0:
                    continue
                for count, val in enumerate(cols):
                    try:
                        info = {'threshold': data['thresholds'].values[0].split(':')[count]}

                        k1 = 'cmp_{:d}_0'.format(count)
                        k2 = 'const_{:d}_0'.format(count)
                        if len(cols) > 1:
                            suggestion, graph_info = plot_and_analyze(total, k1, k2, marked_time, info['threshold'], st,
                                                                      et)
                        else:
                            suggestion, graph_info = plot_and_analyze(total, k1, k2, marked_time, info['threshold'], st,
                                                                      et)

                        # scale the data to make the distance resonable
                        comp = before[k1].astype('float')
                        const = before[k2].astype('float')
                        v1 = comp.max(), comp.min()
                        v2 = const.max(), const.min()
                        maxval = max(v1[0], v2[0])
                        minval = min(v1[1], v2[1])
                        if maxval - minval != 0:
                            comp = comp - minval / (maxval - minval)
                            const = const - minval / (maxval - minval)
                        else:
                            comp.loc[:] = .5
                            const.loc[:] = .5

                        # calculate the distance
                        dist = comp - const
                        dist = np.sqrt(dist * dist).mean()
                        if dist == 0:
                            dist = 999
                        info['distance'] = dist
                        res_series = before['res_{:d}'.format(count)]
                        flop_in_series = get_series_flops(res_series)
                        info['flop_count'] = len(flop_in_series)
                        tp = flop_info.loc[key, 'true_prop']
                        fp = flop_info.loc[key, 'false_prop']

                        # does it match
                        if len(flop_in_series) > 0:
                            info['match'] = True
                        else:
                            if tp > fp:
                                if res_series[0]:
                                    info['match'] = False
                                else:
                                    info['match'] = True
                            else:
                                if res_series[0]:
                                    info['match'] = True
                                else:
                                    info['match'] = False
                        needed_info.append(info)
                    except ValueError as ve:
                        print ve
                plt.show()
            for i in needed_info:
                match_count = 0
                for j in needed_info:
                    if j['threshold'] != i['threshold']:
                        if j['match']:
                            match_count += 1
                s = (i['distance'] + i['flop_count']) / (1 + match_count)
                scores.append((key, i['threshold'], s))

        values = sorted(scores, key=lambda x: x[2])
        for i in values:
            ts = ThreshStatement(i[1], i[0], i[2], suggestion)
            graph_list.append(graph_info)
            thresh_store.import_thresh(ts)


        print marked_time
        thresh_store.sort()
        for i in thresh_store.thresh_list:
            print i
        print '\n\n\n'
        results[marked_time] = thresh_store
    return results


def handle_advance(thresh_info, flops, advances, time_limit=5.0):
    groups = thresh_info.groupby('key')
    results = []
    for marked_time in advances:
        thresh_store = ThreshInfoStore()
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
        # x = filter(lambda arg: arg[1] < time_limit, x)
        # x = sorted(x, key=lambda arg: arg[1])

        # report
        in_limits = thresh_info.between_time(st, et)
        print 'Advance:', marked_time
        if len(x) > 0:
            for flop in x:
                graph_list = []
                windowed_threshold = in_limits[in_limits['key'] == flop[0]]
                size = len([k for k in windowed_threshold.columns if
                            k.startswith('res_') and len(windowed_threshold[k].dropna() > 1)])
                if size > 1:
                    code_thresh = windowed_threshold['thresholds'].values[0].split(':')
                    for i in range(size):
                        stuff = get_series_flops(windowed_threshold.between_time(st, marked_time)['res_{:d}'.format(i)])
                        if len(stuff) > 0:
                            score = (marked_time).total_seconds()
                            k1 = 'cmp_{:d}_0'.format(i)
                            k2 = 'const_{:d}_0'.format(i)
                            sugestion, graph_info = plot_and_analyze(windowed_threshold, k1, k2, marked_time,
                                                                     code_thresh[i], st, et)
                            ts = ThreshStatement(code_thresh[i], key, score, sugestion)
                            thresh_store.import_thresh(ts)
                            graph_list.append(graph_info)
                else:
                    thresh_name = windowed_threshold['thresholds'].values[0]
                    stuff = get_series_flops(windowed_threshold.between_time(st, marked_time)['res_0'])
                    if len(stuff) > 0:
                        score = (marked_time - stuff[-1]).total_seconds()
                        sugestion, graph_info = plot_and_analyze(windowed_threshold, 'cmp_0_0', 'const_0_0',
                                                                 marked_time, thresh_name, st, et)
                        ts = ThreshStatement(thresh_name, key, score, sugestion)
                        thresh_store.import_thresh(ts)
                        graph_list.append(graph_info)
                thresh_store.add_graph(key, graph_list)
        else:
            print 'No flops within {:f}'.format(time_limit)
        thresh_store.sort()
        for i in thresh_store.thresh_list:
            print i
        results[marked_time] = thresh_store
    return results


def plot_and_analyze(windowed_threshold, k1, k2, marked_time, thresh_name, st, et):
    """Add graphics to the graph and also return if the value should be raised or lowered"""
    # get if the value is high or low
    # before
    before = windowed_threshold.between_time(st, marked_time)
    after = windowed_threshold.between_time(marked_time, et)
    diff_b = before[k1].astype('float') - before[k2].astype('float')
    diff_after = after[k1].astype('float') - after[k2].astype('float')
    diff_b = diff_b.mean()
    diff_after = diff_after.mean()
    if diff_b > 0:
        if diff_after > 0:
            sugestion = 'Raise!'
        else:
            sugestion = 'raise'
    else:
        if diff_after < 0:
            sugestion = 'Lower!'
        else:
            sugestion = 'lower'

    idx = windowed_threshold.loc[:, k1].index
    s = [windowed_threshold.loc[:, k1].astype('float').values, windowed_threshold.loc[:, k2].astype('float').values]
    mid = (max(s[0]) + min(s[0])) / 2.0
    snames = ['Compare Value', thresh_name]
    graph_info = GraphInfo(idx, s, snames, [[marked_time], [mid]], sugestion)



    return sugestion, graph_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument('-t', '--thresholds', required=True)
    parser.add_argument('-b', '--bag', required=True)
    args = parser.parse_args()

    thresh = get_thresholds(args.thresholds)
    flops, thresh = get_thresh_info(thresh)
    adv, no_adv = get_times(args.bag)
    no_adv_analysis = handle_no_advance(thresh, flops, no_adv)
    adv_analysis = handle_advance(thresh, flops, adv)
    print no_adv_analysis
    print adv_analysis
    for i in no_adv_analysis.itervalues():
        for j in i.graph_map.iteritems():
            print i[0]
            print i[1]
