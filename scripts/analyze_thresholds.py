#!/usr/bin/env python
# encoding: utf-8
import argparse

import os
import math
import datetime
import sys
from threading import Thread

import pandas as pd
import numpy as np

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import wx
import wx.lib.newevent


# Set up the custom event for background processing of the data
EVT_RESULT_ID = wx.NewId()

MarkSelectedEvent, MARK_SELECTED_EVENT = wx.lib.newevent.NewEvent()
AnalysisResultEvent, ANALYSIS_RESULT_EVENT = wx.lib.newevent.NewEvent()
ThresholdSelected, THRESHOLD_SELECTED_EVENT = wx.lib.newevent.NewEvent()


class ThresholdGraphPanel(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1, size=(50,50))
        self.figure = matplotlib.figure.Figure()
        self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(hbox)

    def update_graphic(self, info_store):
        size = len(info_store)
        self.figure.clear()
        if size == 0:
            self.figure.add_subplot(111)
        else:
            for i in range(size):
                ax = self.figure.add_subplot(size, 1, i+1)
                gi = info_store[i]
                index = gi.index
                for i in zip(gi.names, gi.series):
                    ax.plot(index, i[1], label=i[0])
                a = ax.get_ylim()
                ax_range = a[1] - a[0]
                ax.set_ylim(a[0] - .05 * ax_range, a[1] + .05 * ax_range)
                ax.scatter(gi.scatter_data[0], gi.scatter_data[1], marker='*', c='g', s=40, label='Marked Action')
                ax.text(0.95, 0.01, gi.suggestion, verticalalignment='bottom', horizontalalignment='right',
                        transform=ax.transAxes, color='g', fontsize=16)
                ax.legend()
        self.canvas = FigureCanvas(self, -1, self.figure)


class AnalysisThread(Thread):
    def __init__(self, notify_window, bag_file, thresh_file):
        Thread.__init__(self)
        self._notify_window = notify_window
        self._want_abort = 0
        self.bag_file = bag_file
        self.thresh_file = thresh_file
        self.no_advance = None
        self.advance = None
        self.start()
        self.analysis_results = []

    def run(self):
        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Reading Threshold File'))
        thresh = get_thresholds(self.thresh_file)
        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Calculating Threshold Flops'))
        flops, thresh = get_thresh_info(thresh)
        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Getting user marked information'))
        adv, no_adv = get_times(self.bag_file)
        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Analyzing marked advances'))
        self.no_advance = handle_no_advance(thresh, flops, no_adv)
        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Analyzing marked no advances'))
        self.advance = handle_advance(thresh, flops, adv)

        for i in self.advance.itervalues():
            self.analysis_results.append(i)
        for i in self.no_advance.itervalues():
            self.analysis_results.append(i)

        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Done'))

    def abort(self):
        self._want_abort = 1


class UserMarkPanel(wx.Panel):

    def __init__(self, parent, notify_window):
        wx.Panel.__init__(self, parent)
        self._notify_window = notify_window
        self._list_ctrl = wx.ListCtrl(self, size=(-1, -1), style=wx.LC_REPORT | wx.BORDER_SUNKEN)

        self._list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_item_selected)
        self._list_ctrl.InsertColumn(0, "Type")
        self._list_ctrl.InsertColumn(1, "Time")

        self._row_dict = {}
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._list_ctrl, 1, wx.EXPAND)
        self.SetSizer(hbox)

        self._selected = None

    def add_marks(self, stores):
        # Clear everything out
        self._row_dict.clear()
        self._selected = None
        for idx, res in enumerate(sorted(stores, key=lambda x: x.mtime)):
            self._list_ctrl.InsertStringItem(idx, res.label)
            self._list_ctrl.SetStringItem(idx, 1, str(res.mtime))
            self._row_dict[idx] = res

    def on_item_selected(self, event):
        current_item = event.m_itemIndex
        val = self._row_dict[current_item]
        self._selected = val
        wx.PostEvent(self._notify_window, MarkSelectedEvent(store=val))

    def get_graphical_information(self, threshold_store):
        if self._selected is not None:
            return self._selected.graph_map[threshold_store.stmt_key]
        else:
            return None


class ThresholdInfoPanel(wx.Panel):

    def __init__(self, parent, notify_window):
        wx.Panel.__init__(self, parent)
        self._notify_window = notify_window

        self._list_ctrl = wx.ListCtrl(self, size=(-1, 100), style=wx.LC_REPORT | wx.BORDER_SUNKEN)

        self._list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_item_selected)
        self._list_ctrl.InsertColumn(0, "Key")
        self._list_ctrl.InsertColumn(1, "Threshold")
        self._list_ctrl.InsertColumn(2, "Score")
        self._list_ctrl.InsertColumn(3, "Suggestion")

        self._row_dict = {}

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._list_ctrl, 1, wx.EXPAND)
        self.SetSizer(hbox)
        self._selected = None

    def add_thresholds(self, store):
        # Clear everything out
        if not store.sorted:
            store.sort()
        self._row_dict.clear()
        self._selected = None
        for idx, res in enumerate(store.thresh_list):
            self._list_ctrl.InsertStringItem(idx, res.stmt_key)
            self._list_ctrl.SetStringItem(idx, 1, str(res.threshold))
            self._list_ctrl.SetStringItem(idx, 2, str(res.score))
            self._list_ctrl.SetStringItem(idx, 3, str(res.suggestion))
            self._row_dict[idx] = res


    def on_item_selected(self, event):
        current_item = event.m_itemIndex
        val = self._row_dict[current_item]
        self._selected = val
        wx.PostEvent(self._notify_window, ThresholdSelected(threshold=val))


class ThresholdFrame(wx.Frame):
    def __init__(self, parent, title, thresh_file, bag_file):
        wx.Frame.__init__(self, parent, title=title, size=(1000, 800))

        # which files are we analyzing
        self.thresh_file = thresh_file
        self.bag_file = bag_file

        # set up the split window
        self.left_right = wx.SplitterWindow(self)
        self.top_bottom = wx.SplitterWindow(self.left_right)

        #marking list self argument for which panel to notify -- ie this window
        self.mark_panel = UserMarkPanel(self.left_right, self)

        self.graph_area = ThresholdGraphPanel(self.top_bottom)
        self.thresh_info_area = ThresholdInfoPanel(self.top_bottom, self)

        # set up splitters
        self.left_right.SplitVertically(self.mark_panel, self.top_bottom, 200)
        self.top_bottom.SplitHorizontally(self.graph_area, self.thresh_info_area, 550)
        self.left_right.SetSashGravity(0.0)
        self.top_bottom.SetSashGravity(1.0)

        # set up the status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Hello")

        # handle dealing with the threshold infomation
        ANALYSIS_RESULT_EVENT(self, self.on_result)
        THRESHOLD_SELECTED_EVENT(self, self.on_threshold_selected)
        MARK_SELECTED_EVENT(self, self.on_mark_selected)

    def run_analysis(self):
        self.status_bar.SetStatusText("Begining analysis")
        self.worker = AnalysisThread(self, self.bag_file, self.thresh_file)

    def on_result(self, event):
        self.status_bar.SetStatusText(event.data)
        if event.data == 'Done':
            self.mark_panel.add_marks(self.worker.analysis_results)

    def on_mark_selected(self, event):
        self.thresh_info_area.add_thresholds(event.store)

    def on_threshold_selected(self, event):
        g = self.mark_panel.get_graphical_information(event.threshold)
        self.graph_area.update_graphic(g)



# Analysis classes and functions below here

class GraphInfo(object):
    """Simple class that holds information that is needed to quickly create graphics"""
    def __init__(self, index, series, names, scatter_data, suggestion):
        self.index = index
        self.series = series
        self.names = names
        self.scatter_data = scatter_data
        self.suggestion = suggestion


class ThreshStatement(object):
    """"Contains information about thresholds and statemetns in the source code in one
    location"""
    def __init__(self, threshold, stmt_key, score, suggestion=None):
        self.threshold = threshold
        self.stmt_key = stmt_key
        self.score = score
        self.suggestion = suggestion

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{:f} {:s} {:s}'.format(self.score, self.threshold, self.stmt_key)


class ThreshInfoStore(object):
    """Store that will contain all of the information about one marked time of threshold information"""
    def __init__(self, mtime, advance):
        self.advance = advance
        if advance:
            self.label = 'Action'
        else:
            self.label = 'No Action'
        self.mtime = mtime
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
    """Get the flops in a series of values
    Will return times of every time the series changes from True to False"""
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
    """Get the flops of a key"""
    data_spot = df[df['key'] == key]
    series = data_spot['result']
    return get_series_flops(series)


def get_thresh_info(thresh_df):
    """Get compiled threshold information"""
    keys = []
    trues = []
    falses = []
    totals = []
    pts = []
    pfs = []

    # get flop information as well as number of true and false ect.
    for k, v in thresh_df.groupby('key'):
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
    """Add the times since the last flop to the dataframe"""
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
    """Get a second representation of the timestamp"""
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
        thresh_store = ThreshInfoStore(marked_time,  False)
        et = marked_time + datetime.timedelta(seconds=time_limit)
        st = marked_time - datetime.timedelta(seconds=time_limit)

        for key, data in groups:
            idx = data.index.asof(marked_time)
            needed_info = []
            graph_list = []

            # skip NaN index -> Do not have a record of the threshold at this point in time
            if isinstance(idx, float):
                continue

            before = data.between_time(st, idx)
            total = data.between_time(st, et)
            cols = [x for x in total.columns if x.startswith('res_') and len(total[x].dropna()) > 0]

            # No columns with data at this point
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

                    graph_list.append(graph_info)
                    info['suggestion'] = suggestion
                    # scale the data to make the distance resonable
                    comp = before[k1].astype('float')
                    const = before[k2].astype('float')
                except ValueError as ve:
                    print ve
                    continue
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
            for i in needed_info:
                match_count = 0
                for j in needed_info:
                    if j['threshold'] != i['threshold']:
                        if j['match']:
                            match_count += 1
                s = (i['distance'] + i['flop_count']) / (1 + match_count)
                scores.append((key, i['threshold'], s, i['suggestion']))
            thresh_store.add_graph(key, graph_list)

        values = sorted(scores, key=lambda x: x[2])
        for i in values:
            ts = ThreshStatement(i[1], i[0], i[2], i[3])
            thresh_store.import_thresh(ts)

        thresh_store.sort()
        results[marked_time] = thresh_store
    return results


def handle_advance(thresh_info, flops, advances, time_limit=5.0):
    groups = thresh_info.groupby('key')
    results = {}
    for marked_time in advances:
        thresh_store = ThreshInfoStore(marked_time, True)
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
        if len(x) > 0:
            for flop in x:
                graph_list = []
                windowed_threshold = in_limits[in_limits['key'] == flop[0]]
                size = len([k for k in windowed_threshold.columns if
                            k.startswith('res_') and len(windowed_threshold[k].dropna()) > 1])
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
        thresh_store.sort()
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

    app = wx.App(False)
    frame = ThresholdFrame(None, "Threshold Analysis information", args.thresholds, args.bag)
    frame.run_analysis()
    frame.Show()
    app.MainLoop()


