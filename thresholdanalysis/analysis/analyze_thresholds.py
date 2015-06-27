#!/usr/bin/env python
# encoding: utf-8
import argparse
import json

import os
import datetime
from threading import Thread

import pandas as pd
import numpy as np

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

# noinspection PyUnresolvedReferences
import wx
# noinspection PyUnresolvedReferences
import wx.lib.newevent
# noinspection PyUnresolvedReferences
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin


# Set up the custom event for background processing of the data
EVT_RESULT_ID = wx.NewId()

MarkSelectedEvent, MARK_SELECTED_EVENT = wx.lib.newevent.NewEvent()
AnalysisResultEvent, ANALYSIS_RESULT_EVENT = wx.lib.newevent.NewEvent()
ThresholdSelected, THRESHOLD_SELECTED_EVENT = wx.lib.newevent.NewEvent()

SHOW_CODE_ID = wx.NewId()


class ThresholdGraphPanel(wx.Panel):
    def __init__(self, parent, model):
        wx.Panel.__init__(self, parent, -1)  # ), size=(50, 50))

        self.figure = matplotlib.figure.Figure()
        self.figure.set_facecolor('w')
        self.model = model
        self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.canvas, proportion=1, flag=wx.EXPAND)
        self.SetSizer(hbox)

    def update_graphic(self, result):
        size = len(result.graph_map)
        self.figure.clear()
        if size == 0:
            self.figure.add_subplot(111)
        else:
            highlight = result.highlight
            for i, res in enumerate(result.graph.graph_map):
                info = result.graph.graph_map[res]
                if i == 0:
                    ax = self.figure.add_subplot(size, 1, i + 1)
                    old_ax = ax
                else:
                    # noinspection PyUnboundLocalVariable
                    ax = self.figure.add_subplot(size, 1, i + 1, sharex=old_ax)
                index = result.graph.index
                for series in info['cmp']:
                    ax.plot(index, result.graph.data[series], label=series, linewidth=3, marker='o')
                for series in info['thresh']:
                    ax.plot(index, result.graph.data[series], label=series, linewidth=3)

                a = ax.get_ylim()
                ax_range = a[1] - a[0]
                ax.set_ylim(a[0] - .05 * ax_range, a[1] + .05 * ax_range)
                ax.axvline(x=result.time, linestyle='--', linewidth=2, c='r')
                ax.text(0.95, 0.01, result.graph.suggestions[i], verticalalignment='bottom',
                        horizontalalignment='right',
                        transform=ax.transAxes, color='g', fontsize=16)
                ax.set_title(result.graph.names[i])

                if i != size - 1:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_label('')
                        tick.label.set_visible(False)
                    ax.get_xaxis().set_ticks([])
                else:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(13)
                        # specify integer or one of preset strings, e.g.
                        tick.label.set_rotation(-45)
                    for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(14)
                        # specify integer or one of preset strings, e.g.
                        # ax.legend()
                if i != highlight:
                    alpha = .4
                    # Awesome way to set a property of everything in the axes...
                    for o in ax.findobj(lambda x: hasattr(x, 'set_alpha')):
                        o.set_alpha(alpha)

        # self.canvas.figure = self.figure #= FigureCanvas(self, -1, self.figure)
        self.canvas.draw()

    def clear_graphic(self):
        self.figure.clear()
        self.figure.add_subplot(111)
        self.canvas.draw()


class UserMarkPanel(wx.Panel):
    def __init__(self, parent, notify_window, model):
        wx.Panel.__init__(self, parent)
        self._notify_window = notify_window
        self.model = model
        self._list_ctrl = AutoWidthListCtrl(
            self)  # wx.ListCtrl(self, size=(-1, -1), style=wx.LC_REPORT | wx.BORDER_SUNKEN)

        self._list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_item_selected)
        self._list_ctrl.InsertColumn(0, "Type")
        self._list_ctrl.InsertColumn(1, "Time")

        self._row_dict = {}
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self._list_ctrl, 1, wx.EXPAND)
        self.SetSizer(hbox)

        self._selected = None

    def rebuild_list(self):
        self._row_dict.clear()
        self._list_ctrl.DeleteAllItems()
        self._selected = None
        for idx, res in enumerate(self.model.marks):
            self._list_ctrl.InsertStringItem(idx, res.label)
            self._list_ctrl.SetStringItem(idx, 1, str(res.time))
            self._row_dict[idx] = res

    def on_item_selected(self, event):
        current_item = event.m_itemIndex
        val = self._row_dict[current_item]
        self._selected = val
        wx.PostEvent(self._notify_window, MarkSelectedEvent(index=val))

    def get_graphical_information(self, threshold_store):
        if self._selected is not None:
            return self._selected.graph_map[threshold_store.stmt_key]
        else:
            return None


class AutoWidthListCtrl(wx.ListCtrl, ListCtrlAutoWidthMixin):
    def __init__(self, parent):
        wx.ListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT)
        ListCtrlAutoWidthMixin.__init__(self)


class ThresholdInfoPanel(wx.Panel):
    def __init__(self, parent, notify_window, model):
        wx.Panel.__init__(self, parent)
        self._notify_window = notify_window
        self.model = model

        # self._list_ctrl = wx.ListCtrl(self, size=(-1, 100), style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self._list_ctrl = AutoWidthListCtrl(self)

        self._list_ctrl.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_item_selected)
        self._list_ctrl.InsertColumn(0, "Threshold")
        self._list_ctrl.InsertColumn(1, "Source")
        self._list_ctrl.InsertColumn(2, "Score")
        self._list_ctrl.InsertColumn(3, "Suggestion")
        self._list_ctrl.InsertColumn(4, "Location")

        self._row_dict = {}

        h_box = wx.BoxSizer(wx.HORIZONTAL)
        h_box.Add(self._list_ctrl, 1, wx.EXPAND)
        self.SetSizer(h_box)
        self._selected = None

        self.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.handle_right_click)

    def handle_right_click(self, event):
        menu = wx.Menu()
        menu.Append(SHOW_CODE_ID, "Show Source Code")
        wx.EVT_MENU(menu, SHOW_CODE_ID, self.menu_select_callback)
        self.PopupMenu(menu, event.GetPoint())
        menu.Destroy()

    def menu_select_callback(self, event):
        op = event.GetId()

        if op == SHOW_CODE_ID:
            fname, line = self._selected.stmt_key.split(':')
            if os.path.exists(fname):
                with open(fname) as src_file:
                    lines = src_file.readlines()
                    code = ''.join(lines[int(line) - 3:int(line) + 3])
            else:
                code = 'Could not find file {:s}'.format(fname)
            wx.MessageBox(code, "Source Code", wx.OK)
        else:
            pass

    def add_thresholds(self, index):
        # Clear everything out
        results = self.model.get_results(index)
        results = sorted(results, key=lambda x: x.score)
        self._row_dict.clear()
        self._list_ctrl.DeleteAllItems()
        self._selected = None
        for idx, res in enumerate(results):
            self._list_ctrl.InsertStringItem(idx, str(res.name))
            self._list_ctrl.SetStringItem(idx, 1, str(res.source))
            self._list_ctrl.SetStringItem(idx, 2, str(res.score))
            self._list_ctrl.SetStringItem(idx, 3, str(res.suggestion))
            self._list_ctrl.SetStringItem(idx, 4, str(res.stmt_key))
            self._row_dict[idx] = res

    def on_item_selected(self, event):
        current_item = event.m_itemIndex
        val = self._row_dict[current_item]
        self._selected = val
        wx.PostEvent(self._notify_window, ThresholdSelected(result=val))


class ThresholdFrame(wx.Frame):
    def __init__(self, parent, title, model):
        wx.Frame.__init__(self, parent, title=title, size=(1000, 800))

        # which files are we analyzing
        self.analysis_model = model

        # set up the split window
        self.left_right = wx.SplitterWindow(self)
        self.top_bottom = wx.SplitterWindow(self.left_right)

        # marking list self argument for which panel to notify -- ie this window
        self.mark_panel = UserMarkPanel(self.left_right, self, self.analysis_model)

        self.graph_area = ThresholdGraphPanel(self.top_bottom, self.analysis_model)
        self.thresh_info_area = ThresholdInfoPanel(self.top_bottom, self, self.analysis_model)

        # set up splitters
        self.left_right.SplitVertically(self.mark_panel, self.top_bottom, 200)
        self.top_bottom.SplitHorizontally(self.graph_area, self.thresh_info_area, 550)
        self.left_right.SetSashGravity(0.0)
        self.top_bottom.SetSashGravity(1.0)
        self.worker = None

        # set up the status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Hello")

        # handle dealing with the threshold infomation
        ANALYSIS_RESULT_EVENT(self, self.on_result)
        THRESHOLD_SELECTED_EVENT(self, self.on_threshold_selected)
        MARK_SELECTED_EVENT(self, self.on_mark_selected)

    def run_analysis(self):
        self.status_bar.SetStatusText("Begining analysis")
        self.worker = AnalysisThread(self, self.analysis_model)

    def on_result(self, event):
        self.status_bar.SetStatusText(event.data)
        if event.data == 'Done':
            self.mark_panel.rebuild_list()

    def on_mark_selected(self, event):
        self.graph_area.clear_graphic()
        self.thresh_info_area.add_thresholds(event.index)

    def on_threshold_selected(self, event):
        self.graph_area.update_graphic(event.result)


class AnalysisThread(Thread):
    def __init__(self, notify_window, analysis_model):
        Thread.__init__(self)
        self._notify_window = notify_window
        self._want_abort = 0
        self.model = analysis_model
        self.no_advance = None
        self.advance = None
        self.start()
        self.analysis_results = []

    def run(self):
        self.model.load_mark_times()
        self.model.load_data()
        self.model.get_flop_information()
        self.model.compute_results()
        wx.PostEvent(self._notify_window, AnalysisResultEvent(data='Done'))

    def abort(self):
        self._want_abort = 1


class ThresholdBackgroundNode(object):
    def __init__(self):
        self.dirty = False


def take2(arr):
    s = 0
    n = s + 2
    while n <= len(arr):
        v = arr[s:n]
        s = n
        n = s + 2
        yield v


class StaticInfoMap(object):
    """Class that handles everything that has to do with the static
    info that is produced by parsing and threshold identification process"""

    def __init__(self, file_map):
        self._info = {}
        if file_map is not None:
            for k, f in take2(file_map):
                self._load_file_info(k, f)

    def get_static_info(self, key_name):
        if key_name in self._info:
            return self._info[key_name]
        else:
            self._load_file_info(key_name, os.path.splitext(key_name)[0] + '_thresh_info.json')
            return self._info[key_name]

    def _load_file_info(self, key_name, file_name=None):
        if file_name is None:
            file_name = key_name
        if file_name not in self._info:
            with open(file_name, 'r') as open_f:
                info = json.load(open_f)
                self._info[key_name] = info

    def get_max_distance(self):
        max_val = 0
        for i in self._info.itervalues():
            for j in i.itervalues():
                if int(j['distance']) > 0:
                    max_val = int(j['distance'])
        return max_val


class UserMark(object):
    def __init__(self, time, isaction):
        self.time = time
        self.isaction = isaction
        if isaction:
            self.label = 'Action'
        else:
            self.label = 'No Action'

    def is_advance(self):
        return self.isaction

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.isaction:
            return str(self.time) + ' ' + 'Action'
        else:
            return str(self.time) + ' ' + 'No Action'


class InvalidException(Exception):
    def __init__(self, key):
        Exception.__init__(self)
        self.message = key + ' is invalid'


class ResultStore(object):
    def __init__(self):
        self._results = {}
        self._valid = {}

    def add_result(self, key, results):
        self._results[key] = results
        self._valid[key] = True

    def get_result(self, key):
        if not self._valid[key]:
            raise InvalidException(key)
        else:
            return self._results[key]

    def invalidate(self):
        for i in self._valid.keys():
            self._valid[i] = False

    def check_valid(self, key):
        if key in self._valid:
            return self._valid[key]
        else:
            return False


class GraphStorage(object):
    def __init__(self):
        self.index = None
        self.data = {}
        self.map = {}
        self.names = []
        self.suggestions = []


class ThresholdAnalysisModel(object):
    """New model to hold all of the analysis information for the gui tool.
        This allows easy linking of all of the calls and the ability to place all
        of the data in one easy to access location instead of accross multiple classes
        that make adding and editing stuff a pain."""

    def __init__(self, bag_file=None, thresh_file=None, file_map=None, master_window=None, live=False):
        self._bag_file = bag_file
        self._thresh_file = thresh_file

        self._static_info = StaticInfoMap(file_map)
        self.marks = []

        self.analysis_parameters = {'action_time_limit': 5.0, 'no_action_time_limit': 3.0, 'graph_time_limit': 5.0}
        self.result_store = ResultStore()

        self._thresh_df = None
        self.summary_df = None

        self.marked_actions = None
        self.marked_results = None
        self.marked_results = None

        self.marked_no_actions = None
        self.result_dict = {}
        self.compiled_results = []
        self.live = live
        if self.live:
            self.background_node = ThresholdBackgroundNode()

        self.advanced_results = None
        self.no_advanced_results = None

        # wx notification stuff
        self._notify_window = master_window

    def set_notify_window(self, window):
        """set the notification window"""
        self._notify_window = window

    def post_notification(self, notification):
        if self._notify_window is not None:
            wx.PostEvent(self._notify_window, AnalysisResultEvent(data=notification))

    def load_data(self):
        self.post_notification('Loading data')
        if self._thresh_file is not None:
            self._thresh_df = pd.read_csv(self._thresh_file, parse_dates=True, index_col=0)
            for key in self._thresh_df.key.unique():
                self._static_info.get_static_info(key.split(':')[0])
        else:
            self._thresh_df = pd.DataFrame()

        self.post_notification('Done Loading Data')

    def get_flop_information(self):
        if self._thresh_df is None:
            raise Exception('Threshold File not loaded')

        self.post_notification('Calculating Flops and stuff')
        """Get compiled threshold information"""
        keys = []
        trues = []
        falses = []
        totals = []
        pts = []
        pfs = []

        # get flop information as well as number of true and false ect.
        for k, v in self._thresh_df.groupby('key'):
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
            if t + f == 0:
                pt = 0
                pf = 0
            else:
                pt = (t / (float(t + f)))
                pf = (f / (float(t + f)))
            pts.append(pt)
            pfs.append(pf)

        df = pd.DataFrame(data={'true_count': trues, 'false_count': falses, 'count': totals,
                                'true_prop': pts, 'false_prop': pfs}, index=keys)

        # create the dataframe
        # now add flop_times
        self._thresh_df['last_flop'] = np.NaN
        for i in df.index:
            flop_locations = get_flops(self._thresh_df, i)
            self._thresh_df = add_times(self._thresh_df, i, flop_locations)
        self.summary_df = df
        self.post_notification('Done calculating flops')

    def load_mark_times(self):
        self.post_notification('Finding user marks')
        if self._bag_file is None:
            self.marks = []
            return
        bag_df = pd.read_csv(self._bag_file, parse_dates=True, index_col=0)
        marks = []
        if 'mark_no_action__data_nsecs' in bag_df.columns:
            idx = bag_df.mark_no_action__data_nsecs.dropna().index
            vals = bag_df.loc[idx, ['mark_no_action__data_secs', 'mark_no_action__data_nsecs']]
            for _, data in vals.iterrows():
                s = data['mark_no_action__data_secs']
                ns = data['mark_no_action__data_nsecs']
                time = s + ns / 1000000000.0
                time = pd.to_datetime(time, unit='s')
                marks.append(UserMark(time, False))
        if 'mark_action__data_nsecs' in bag_df.columns:
            idx = bag_df.mark_action__data_nsecs.dropna().index
            vals = bag_df.loc[idx, ['mark_action__data_secs', 'mark_action__data_nsecs']]
            for _, data in vals.iterrows():
                s = data['mark_action__data_secs']
                ns = data['mark_action__data_nsecs']
                time = s + ns / 1000000000.0
                time = pd.to_datetime(time, unit='s')
                marks.append(UserMark(time, True))
        self.marks = sorted(marks, key=lambda x: x.time)
        self.post_notification('Done finding user marks')

    def compute_results(self):
        for i in self.marks:
            if i.isaction:
                res = self.get_advance_results(i)
                self.result_store.add_result(i, res)
            else:
                res = self.get_no_advance_results(i)
                self.result_store.add_result(i, res)

    def get_advance_results(self, mark):
        self.post_notification("Computing advance result")
        results = []

        # compute time limits and the like
        time_limit = self.analysis_parameters['action_time_limit']
        time = mark.time
        st = time - datetime.timedelta(seconds=time_limit)
        et = time + datetime.timedelta(seconds=time_limit)

        # get data
        thresh_df = self.get_thresh_df().between_time(st, et)
        calc_data = self.get_thresh_df().between_time(st, time)
        calc_groups = calc_data.groupby('key')
        elapsed_times = {}
        for key, data in calc_groups:
            # calculate how long it has been since the last flop
            lf = data.tail(1)['last_flop'][0]
            tdelta = time - data.tail(1).index[0]
            elapsed = (lf + tdelta).total_seconds()
            elapsed_times[key] = elapsed

        # get maximum time to last flop...
        # max_time = max([x for x in elapsed_times.itervalues()])
        groups = thresh_df.groupby('key')
        for key, data in groups:
            elapsed = elapsed_times[key]
            if elapsed < time_limit:

                # get static information
                f_name, lineno = key.split(':')
                thresh_information = self._static_info.get_static_info(f_name)[key]
                threshs = thresh_information['thresh']
                names = thresh_information['names']
                srcs = thresh_information['sources']
                comparisions = thresh_information['comparisons']

                # enumerate all of the thresholds and comparisons in the file..
                graph = GraphStorage()
                graph_map = {i['res']: i for i in thresh_information['comparisons']}
                graph.graph_map = graph_map

                for idx, comp in enumerate(comparisions):
                    res = comp['res']
                    if len(comp['thresh']) > 1:
                        print 'Error Cannot handle multiple thresh in one comparision...'

                    # now get the flops for each individual result in this thing
                    flops = get_series_flops(data.between_time(st, time)[res])
                    if len(flops) > 0:
                        elapsed = (time - flops[-1]).total_seconds()

                    # get score... For now super simple elapsed time
                    score = elapsed

                    # Compute the suggestion...still needs work
                    sugestion = self.get_suggestion(time, data, comp['cmp'][0],  comp['thresh'][0], res,  True)

                    # build up the result
                    one_result = AnalysisResult()

                    # store information
                    one_result.threshold = comp['thresh'][0]
                    thresh_idx = threshs.index(one_result.threshold)
                    one_result.source = srcs[thresh_idx]
                    one_result.name = names[thresh_idx]
                    one_result.score = score
                    one_result.suggestion = sugestion
                    one_result.time = time
                    one_result.stmt_key = key
                    one_result.highlight = idx

                    one_result.graph_map = graph_map
                    one_result.graph = graph
                    graph_data = data.between_time(st, time + datetime.timedelta(seconds=time_limit))
                    graph.index = graph_data.index
                    graph.names.append(names[thresh_idx])
                    graph.suggestions.append(sugestion)
                    for i in graph_map[res]['cmp']:
                        graph.data[i] = graph_data[i].values
                    for i in graph_map[res]['thresh']:
                        graph.data[i] = graph_data[i].values
                    results.append(one_result)

        self.post_notification("Done with advance result")
        return results

    def get_suggestion(self, time, data, comp_key, thresh_key, res_key, action):
        """Get a suggestion based on other values"""
        suggestion = ''

        #TODO this still needs some work..
        if action:
            lf = get_series_flops(data.between_time(time - datetime.timedelta(seconds=self.analysis_parameters['no_action_time_limit']), time)[res_key])
            if len(lf) == 0:
                comp = data[comp_key]
                thresh = data[thresh_key]
                over = len(comp[comp < thresh])
                under = len(comp) - over
                if over > under:
                    suggestion = 'Raise'
                else:
                    suggestion = 'Lower'
            else:
                comp = data.loc[lf[-1], comp_key]
                thresh = data.loc[lf[-1], thresh_key]
                if comp > thresh:
                    suggestion = 'Raise'
                else:
                    suggestion = 'Lower'
        else:
            lf = get_series_flops(data[res_key])
            if len(lf) > 0:
                print 'asdfjaslj'
                data = data.between_time(lf[-1], time)
            comp = data[comp_key]
            thresh = data[thresh_key]
            above = len(comp[comp.values < thresh.values])
            below = len(comp[comp.values > thresh.values])
            if above > below:
                suggestion = 'Lower'
            else:
                suggestion = 'Raise'

        return suggestion


    def get_no_advance_results(self, mark):
        results = []

        # compute time limits and the like
        time_limit = self.analysis_parameters['no_action_time_limit']
        graph_limit = self.analysis_parameters['graph_time_limit']
        time = mark.time
        # get data
        thresh_df = self.get_thresh_df()
        maxes = thresh_df.groupby('key').max()
        mins = thresh_df.groupby('key').min()
        for key, data in thresh_df.groupby('key'):
            calc_data = data.between_time(time - datetime.timedelta(seconds=time_limit), time)
            # get static information
            f_name, lineno = key.split(':')
            thresh_information = self._static_info.get_static_info(f_name)[key]
            threshs = thresh_information['thresh']
            names = thresh_information['names']
            srcs = thresh_information['sources']
            ress = thresh_information['res']
            comparisions = thresh_information['comparisons']

            # enumerate all of the thresholds and comparisions in the file..
            graph = GraphStorage()
            graph_map = {i['res']: i for i in thresh_information['comparisons']}
            graph.graph_map = graph_map

            # calculate scores here
            for idx, comp in enumerate(comparisions):
                if len(comp['thresh']) > 1:
                    print 'Error Cannot handle multiple thresh in one comparision...'
                t = comp['thresh'][0]
                c = comp['cmp'][0]
                res = comp['res']
                maxval = max(maxes.loc[key, t], maxes.loc[key, c])
                minval = min(mins.loc[key, t], mins.loc[key, c])
                cseries = calc_data[c]
                const = data[t]
                if maxval - minval != 0:
                    cseries = (cseries - minval) / (maxval - minval)
                    const = (const - minval) / (maxval - minval)
                else:
                    cseries.loc[:] = .5
                    const.loc[:] = .5
                dist = cseries - const
                dist = np.sqrt(dist * dist).mean()
                if dist == 0:
                    dist = 999
                flop_in_series = len(get_series_flops(calc_data[res]))
                flop_in_series += 1
                different = 0

                # calculate the number of different values on each of the other thresholds.
                if thresh_information['num_comparisons'] > 1:
                    comptype = thresh_information['opmap'][thresh_information['opmap'].keys()[0]]['op']
                    val_here = calc_data[res].tail(1).values[0]
                    for r in ress:
                        if r != res and comptype == 'and':
                            last_value = calc_data[r].tail(1)
                            if val_here != last_value:
                                different += 1
                # get rid of dividing by zero problem
                different += 1
                d = thresh_information['distance']
                md = self._static_info.get_max_distance()

                num_comparisions = float(thresh_information['num_comparisons'])
                if num_comparisions == 0:
                    num_comparisions = 1
                d_score = d / float(md)
                if d_score == 0:
                    d_score = 1
                score = (dist / d_score) * flop_in_series * (different / num_comparisions)
                # s = (i['distance'] + i['flop_count']) / (1 + match_count)

                suggestion = self.get_suggestion(time, calc_data, comp['cmp'], comp['thresh'], res, False)

                one_result = AnalysisResult()

                one_result.threshold = comp['thresh'][0]
                thresh_idx = threshs.index(one_result.threshold)
                one_result.source = srcs[thresh_idx]
                one_result.name = names[thresh_idx]
                one_result.score = score
                one_result.suggestion = suggestion
                one_result.time = time
                one_result.stmt_key = key
                one_result.highlight = idx

                one_result.graph_map = graph_map
                one_result.graph = graph
                graph_data = data.between_time(time - datetime.timedelta(seconds=graph_limit),
                                               time + datetime.timedelta(seconds=graph_limit))
                graph.index = graph_data.index
                graph.names.append(names[thresh_idx])
                graph.suggestions.append(suggestion)
                for i in graph_map[res]['cmp']:
                    graph.data[i] = graph_data[i].values
                for i in graph_map[res]['thresh']:
                    graph.data[i] = graph_data[i].values
                results.append(one_result)

        self.post_notification("Done with advance result")
        return results

    def get_results(self, index):
        return self.result_store.get_result(index)

    def get_thresh_df(self):
        # TODO Add import of new information from background runtime
        return self._thresh_df


class AnalysisResult(object):
    """Object to hold results"""
    def __init__(self):
        """Init with all of the data values it stores set to none"""
        self.graph_index = None
        self.graph_data = {}
        self.threshold = None
        self.source = None
        self.name = None
        self.score = None
        self.suggestion = None
        self.time = None
        self.stmt_key = None
        self.highlight = None
        self.graph_map = None
        self.graph = None


def get_series_flops(series):
    """Get the flops in a series of values
    Will return times of every time the series changes from True to False"""
    a = series.dropna()
    fvals = a[a != a.shift(1)]
    fvals = fvals.index[1:]
    return fvals


def get_flops(df, key):
    """Get the flops of a key"""
    data_spot = df[df['key'] == key]
    series = data_spot['result']
    return get_series_flops(series)


def add_times(to_add_df, key, locations):
    """Add the times since the last flop to the dataframe"""
    vals = to_add_df[to_add_df['key'] == key]
    cur_time = vals.index.to_series()
    series = pd.Series(data=locations, index=locations)
    series = series.reindex(cur_time.index)
    s = (cur_time - series.ffill())
    to_add_df.ix[s.index, 'last_flop'] = s
    return to_add_df


def ts_to_sec(ts, epoch=datetime.datetime.utcfromtimestamp(0)):
    """Get a second representation of the timestamp"""
    delta = ts - epoch
    try:
        return delta.total_seconds()
    except:
        return np.NaN


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument('-t', '--thresholds', required=True)
    parser.add_argument('-b', '--bag', required=True)
    parser.add_argument('-k', '--key_map', nargs='*')
    args = parser.parse_args()

    tam = ThresholdAnalysisModel(args.bag, args.thresholds, args.key_map)

    app = wx.App(False)
    frame = ThresholdFrame(None, "Threshold Analysis information", tam)
    frame.run_analysis()
    frame.Show()
    app.MainLoop()
