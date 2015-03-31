#!/usr/bin/env python
# encoding: utf-8
import argparse
import json
import glob

import os
import datetime

from threading import Thread

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import pandas as pd
import numpy as np

# noinspection PyUnresolvedReferences
import wx
# noinspection PyUnresolvedReferences
import wx.lib.newevent
# noinspection PyUnresolvedReferences
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
from threshold_node import ThresholdNode


# Set up the custom event for background processing of the data
EVT_RESULT_ID = wx.NewId()
TIMER_ID = wx.NewId()

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
        self.figure.clear()

        ax = self.figure.add_subplot(1, 1, 1)
        ax = self.figure.add_subplot(1, 1, 1)
        index = result.graph.index
        ax.plot(index, result.graph.cmp, label='cmp', linewidth=3, marker='o')
        ax.plot(index, result.graph.thresh, label='thresh', linewidth=3)
        a = ax.get_ylim()
        ax_range = a[1] - a[0]
        ax.set_ylim(a[0] - .05 * ax_range, a[1] + .05 * ax_range)
        ax.axvline(x=result.time, linestyle='--', linewidth=2, c='r')
        ax.text(0.95, 0.01, result.graph.suggestion, verticalalignment='bottom',
                horizontalalignment='right',
                transform=ax.transAxes, color='g', fontsize=16)
        ax.set_title(result.graph.name)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
            # specify integer or one of preset strings, e.g.
            tick.label.set_rotation(-45)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
            # specify integer or one of preset strings, e.g.
            # ax.legend()
        # self.canvas.figure = self.figure #= FigureCanvas(self, -1, self.figure)
        self.canvas.draw()

    def clear_graphic(self):
        self.figure.clear()
        self.figure.add_subplot(111)
        self.canvas.draw()


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


class ThresholdPairPanel(wx.Panel):
    def __init__(self, parent, notify_window, model):
        wx.Panel.__init__(self, parent)
        self._notify_window = notify_window
        self.model = model
        self._list_ctrl = AutoWidthListCtrl(self)
        self._list_ctrl.InsertColumn(0, "Type")
        self._list_ctrl.InsertColumn(1, "Source")
        self._list_ctrl.InsertColumn(2, "Value")
        self._list_ctrl.InsertColumn(3, "Suggestion")
        for i in range(self._list_ctrl.GetColumnCount()):
            self._list_ctrl.SetColumnWidth(i, -2)

        self._row_dict = {}
        self._data_dict = {}
        hbox = wx.BoxSizer(wx.VERTICAL)
        hbox.Add(self._list_ctrl, 1, wx.EXPAND)
        self.SetSizer(hbox)
        self._selected = None

    def rebuild_list(self):
        self._row_dict.clear()
        self._data_dict.clear()
        self._list_ctrl.DeleteAllItems()
        self._selected = None

        values = self.model.get_threshold_information()
        for idx, res in enumerate(values):
            self._list_ctrl.InsertStringItem(idx, res[1])
            self._list_ctrl.SetStringItem(idx, 1, str(res[2]))
            self._list_ctrl.SetStringItem(idx, 2, str(res[3]))
            self._list_ctrl.SetStringItem(idx, 3, "")
            self._row_dict[idx] = res
            self._data_dict[res[0]] = res
        for i in range(self._list_ctrl.GetColumnCount()):
            self._list_ctrl.SetColumnWidth(i, -1)
        self._list_ctrl.SetColumnWidth(2, -2)

    def mark_possible(self, idx, take=2):
        results = self.model.get_results(idx)
        results = sorted(results, key=lambda x: x.score)
        # now either have a cutoff score or cutoff number
        results = results[:take]
        self._selected = []
        # Clear old text
        for r, _ in self._row_dict.iteritems():
            self._list_ctrl.SetStringItem(r, 3, "")

        # add new text
        for idx, res in enumerate(results):
            val = self._data_dict[res.stmt_key]
            for r, v in self._row_dict.iteritems():
                if v == val:
                    self._selected.append(r)
                    self._list_ctrl.SetStringItem(r, 3, res.suggestion)

        # set selection
        for r, _ in self._row_dict.iteritems():
            if r in self._selected:
                self._list_ctrl.SetItemState(r, wx.LIST_STATE_SELECTED, wx.LIST_STATE_SELECTED)
            else:
                self._list_ctrl.SetItemState(r, 0, wx.LIST_STATE_SELECTED)


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


class ThresholdFrame(wx.Frame):
    def __init__(self, parent, title, model):
        wx.Frame.__init__(self, parent, title=title, size=(900, 700))

        # which files are we analyzing
        self.analysis_model = model

        # set up large split window
        self.left_right = wx.SplitterWindow(self)

        # Add the user mark panel and the notebook
        self.mark_panel = UserMarkPanel(self.left_right, self, self.analysis_model)
        self.notebook = wx.Notebook(self.left_right)

        # marking list self argument for which panel to notify -- ie this window
        self.thresh_pair_panel = ThresholdPairPanel(self.notebook, self, self.analysis_model)

        # set up splitters

        self.graph_page = wx.SplitterWindow(self.notebook)

        self.graph_area = ThresholdGraphPanel(self.graph_page, self.analysis_model)
        self.thresh_info_area = ThresholdInfoPanel(self.graph_page, self, self.analysis_model)

        # add to notebook
        self.notebook.AddPage(self.thresh_pair_panel, "Threshold View")
        self.notebook.AddPage(self.graph_page, "Graphs")

        self.left_right.SplitVertically(self.mark_panel, self.notebook, 200)
        self.left_right.SetSashGravity(0.0)
        self.graph_page.SplitHorizontally(self.graph_area, self.thresh_info_area, 550)
        self.graph_page.SetSashGravity(.8)
        self.worker = None

        # set up the status bar
        self.status_bar = self.CreateStatusBar()
        self.status_bar.SetStatusText("Hello")

        # handle dealing with the threshold infomation
        ANALYSIS_RESULT_EVENT(self, self.on_result)
        THRESHOLD_SELECTED_EVENT(self, self.on_threshold_selected)
        MARK_SELECTED_EVENT(self, self.on_mark_selected)

        self.timer = wx.Timer(self, TIMER_ID)
        self.timer.Start(1000)
        wx.EVT_TIMER(self, TIMER_ID, self.on_timer)

    def on_timer(self, _):
        if self.analysis_model is not None:
            rebuild = self.analysis_model.process_new_marks()
            if rebuild:
                self.mark_panel.rebuild_list()
                self.thresh_pair_panel.rebuild_list()

    def on_result(self, event):
        self.status_bar.SetStatusText(event.data)
        if event.data == 'Done':
            self.mark_panel.rebuild_list()
            self.thresh_pair_panel.rebuild_list()

    def on_mark_selected(self, event):
        self.graph_area.clear_graphic()
        self.thresh_info_area.add_thresholds(event.index)
        self.thresh_pair_panel.mark_possible(event.index)

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

    def abort(self):
        self._want_abort = 1


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

    def __init__(self, file_map=None, directory=None):
        self._info = {}
        if file_map is not None and directory is not None:
            print("ERROR cannot both be none skipping!")
        if file_map is not None:
            for k, f in take2(file_map):
                self._load_file_info(k, f)
        if directory is not None:
            if os.path.exists(directory):
                if not directory[-1] == '/':
                    directory += '/'
                for f in glob.glob(directory + "*.json"):
                    self._load_file_info(f, f)
            else:
                print("ERROR directory does not exist")

    def get_key_type_src(self):
        ret_val = []
        for i in self._info:
            ty = self._info[i]['type']
            value = self._info[i]['source']
            ret_val.append((i, ty, value))
        return ret_val

    def get_static_info(self, key_name):
        if key_name in self._info:
            return self._info[key_name]
        else:
            print("does not exist...trying to load file: {:s}".format(key_name))
            self._load_file_info(key_name, os.path.splitext(key_name)[0] + '_thresh_info.json')
            return self._info[key_name]

    def _load_file_info(self, key_name, file_name=None):
        if file_name is None:
            file_name = key_name
        if file_name not in self._info:
            with open(file_name, 'r') as open_f:
                info = json.load(open_f)
                for i in info:
                    self._info[i] = info[i]

    def get_max_distance(self):
        max_val = 0
        for i in self._info.itervalues():
            if int(i['distance']) > 0:
                max_val = int(i['distance'])
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
        self.index = []
        self.cmp = []
        self.thresh = []
        self.name = ''
        self.suggestion = ''


class ThresholdAnalysisModel(object):
    """New model to hold all of the analysis information for the gui tool.
        This allows easy linking of all of the calls and the ability to place all
        of the data in one easy to access location instead of accross multiple classes
        that make adding and editing stuff a pain."""

    def __init__(self, bag_record=None, mark_file=None, thresh_file=None, file_map=None, info_directory=None,
                 master_window=None, is_live=True, namespace=None):
        self.background_node = ThresholdNode(is_live)
        if bag_record is not None:
            self.background_node.import_bag_file(bag_record, namespace)
        else:
            if mark_file:
                self.background_node.import_mark_file(mark_file)
            if thresh_file:
                self.background_node.import_thresh_file(thresh_file)

        self._static_info = StaticInfoMap(file_map, info_directory)
        self.marks = []
        self.analysis_parameters = {'action_time_limit': 5.0, 'no_action_time_limit': 3.0, 'graph_time_limit': 5.0}
        self.result_store = ResultStore()
        self._thresh_df = pd.DataFrame()
        self.marked_actions = None
        self.marked_results = None
        self.marked_results = None
        self.marked_no_actions = None
        self.result_dict = {}
        self.compiled_results = []
        self.advanced_results = None
        self.no_advanced_results = None
        # wx notification stuff
        self._notify_window = master_window

    def get_threshold_information(self):
        ret_vals = []
        for i in self._static_info.get_key_type_src():
            key = i[0]
            ty = i[1]
            name = i[2]
            value = self.get_thresh_value(key)
            ret_vals.append((key, ty, name, value))
        return ret_vals

    def get_thresh_value(self, key):
        df = self.get_thresh_df()
        if len(df) == 0:
            return np.NaN
        temp = df[df['key'] == key].tail(1)
        if len(temp) > 0:
            return temp['thresh'].values[0]
        else:
            return np.NaN

    def set_notify_window(self, window):
        """set the notification window"""
        self._notify_window = window

    def post_notification(self, notification):
        if self._notify_window is not None:
            wx.PostEvent(self._notify_window, AnalysisResultEvent(data=notification))

    def process_new_marks(self):
        new_marks = [UserMark(*x) for x in self.background_node.get_new_marks()]
        # Check to see if we need to rebuild the model and all of that good stuff
        if len(new_marks) > 0:
            self.marks.extend(new_marks)
            self.compute_results(new_marks)
            return True
        return False

    def compute_results(self, marks):
        # rebuild dataframe here.
        self.rebuild_dataframe()
        for i in marks:
            if i.isaction:
                res = self.get_advance_results(i)
                self.result_store.add_result(i, res)
            else:
                res = self.get_no_advance_results(i)
                self.result_store.add_result(i, res)

    def rebuild_dataframe(self):
        data = self.background_node.get_new_threshold_data()
        self._thresh_df = self._thresh_df.append(data)

    def get_advance_results(self, mark):
        results = []
        # compute time limits and the like
        time_limit = self.analysis_parameters['action_time_limit']
        time = mark.time
        st = time - datetime.timedelta(seconds=time_limit)
        et = time + datetime.timedelta(seconds=time_limit)

        # get data
        thresh_df = self.get_thresh_df()
        if len(thresh_df) == 0:
            return results
        thresh_df = thresh_df.between_time(st, et)
        calc_data = self.get_thresh_df().between_time(st, time)
        calc_groups = calc_data.groupby('key')
        elapsed_times = {}
        for key, data in calc_groups:
            # calculate how long it has been since the last flop
            try:
                lf = data.tail(1)['last_cmp_flop'][0]
                tdelta = (time - lf).total_seconds()
                elapsed_times[key] = tdelta
            except:
                elapsed_times[key] = 99999
        # get maximum time to last flop...
        # max_time = max([x for x in elapsed_times.itervalues()])
        groups = thresh_df.groupby('key')
        for key, data in groups:
            elapsed = elapsed_times[key]
            if elapsed < time_limit:
                score = elapsed
                sugestion = self.get_suggestion(time, data, 'cmp', 'thresh', 'res', True)
                # build up the result
                one_result = AnalysisResult()
                thresh_information = self._static_info.get_static_info(key)

                # store information
                one_result.threshold = thresh_information['name']
                one_result.source = thresh_information['source']
                one_result.name = thresh_information['name']
                one_result.score = score
                one_result.suggestion = sugestion
                one_result.time = time
                one_result.stmt_key = key

                graph = GraphStorage()
                one_result.graph = graph
                graph_data = data.between_time(st, time + datetime.timedelta(seconds=time_limit))
                graph.index = graph_data.index
                graph.name = thresh_information['source']
                graph.suggestion = sugestion
                graph.cmp = graph_data['cmp'].values
                graph.thresh = graph_data['thresh'].values
                results.append(one_result)

        self.post_notification("Done with advance result")
        return results

    def get_suggestion(self, time, data, comp_key, thresh_key, res_key, action):
        """Get a suggestion based on other values"""
        # TODO this still needs some work..
        if action:
            lf = get_series_flops(
                data.between_time(time - datetime.timedelta(seconds=self.analysis_parameters['no_action_time_limit']),
                                  time)[res_key])
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
        if len(self.get_thresh_df()) == 0:
            print 'empty data set'
            return results
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
            thresh_information = self._static_info.get_static_info(key)
            graph = GraphStorage()
            maxval = max(maxes.loc[key, 'thresh'], maxes.loc[key, 'cmp'])
            minval = min(mins.loc[key, 'thresh'], mins.loc[key, 'cmp'])
            cseries = calc_data['cmp']
            const = data['thresh']
            if np.isnan(maxval) or np.isnan(minval):
                cseries[:] = .5
                const[:] = .5
            elif maxval - minval != 0:
                cseries = (cseries - minval) / (maxval - minval)
                const = (const - minval) / (maxval - minval)
            else:
                cseries = pd.Series(data=.5, index=cseries.index)
                const = pd.Series(data=.5, index=const.index)
            dist = cseries - const
            dist = np.sqrt(dist * dist).mean()
            if dist == 0:
                dist = 999
            flop_in_series = len(calc_data[calc_data['flop'] == True])

            # TODO Take into account other thresholds?
            # calculate the number of different values on each of the other thresholds.

            d = thresh_information['distance']
            md = self._static_info.get_max_distance()
            num_comparisions = float(thresh_information['other_thresholds'])
            if num_comparisions == 0:
                num_comparisions = 1
            d_score = d / float(md)
            if d_score == 0:
                d_score = 1
            score = (dist / d_score) + flop_in_series  # / num_comparisions)

            suggestion = self.get_suggestion(time, calc_data, 'cmp',
                                             'thresh', 'res', False)

            one_result = AnalysisResult()

            one_result.threshold = thresh_information['source']
            one_result.source = thresh_information['source']
            one_result.name = thresh_information['source']
            one_result.score = score
            one_result.suggestion = suggestion
            one_result.time = time
            one_result.stmt_key = key
            one_result.highlight = 0

            one_result.graph = graph
            graph_data = data.between_time(time - datetime.timedelta(seconds=graph_limit),
                                           time + datetime.timedelta(seconds=graph_limit))
            graph.index = graph_data.index
            graph.name = thresh_information['name']
            graph.suggestion = suggestion
            graph.cmp = graph_data['cmp'].values
            graph.thresh = graph_data['thresh'].values
            results.append(one_result)
        return results

    def get_results(self, index):
        return self.result_store.get_result(index)

    def get_thresh_df(self):
        return self._thresh_df


class AnalysisResult(object):
    """Object to hold results"""

    def __init__(self):
        """Init with all of the data values it stores set to `none"""
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


# Flopping stuff
def get_flop_information(df):
    """Add last flop information to a dataframe"""
    if df is None:
        raise Exception('Threshold File not loaded')

    # Get compiled threshold information
    keys = []
    trues = []
    falses = []
    totals = []
    pts = []
    pfs = []

    # no data
    if len(df) == 0:
        return df

    for k, v in df.groupby('key'):
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

    summary_df = pd.DataFrame(data={'true_count': trues, 'false_count': falses, 'count': totals,
                                    'true_prop': pts, 'false_prop': pfs}, index=keys)

    # create the dataframe
    # now add flop_times
    df.loc[:, 'last_flop'] = np.NaN
    for i in summary_df.index:
        flop_locations = get_flops(df, i)
        df = add_times(df, i, flop_locations)
    return df


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
    parser.add_argument('-t', '--thresholds', )
    parser.add_argument('-m', '--mark_file', )
    parser.add_argument('-b', '--bag_record', )
    parser.add_argument('-k', '--key_map', nargs='*', )
    parser.add_argument('-d', '--info_directory', )
    parser.add_argument('--not_live', action='store_true')
    parser.add_argument('--namespace', )
    parser.add_argument('rest', nargs='*')
    args = parser.parse_args()

    live = True
    if args.not_live or args.thresholds is not None or args.mark_file is not None or args.bag_record is not None:
        print 'running on old records not live!'
        live = False

    # create the model
    tam = ThresholdAnalysisModel(mark_file=args.mark_file, thresh_file=args.thresholds, file_map=args.key_map,
                                 info_directory=args.info_directory, bag_record=args.bag_record,
                                 namespace=args.namespace, is_live=live)

    app = wx.App(False)
    frame = ThresholdFrame(None, "Threshold Analysis information", tam)
    frame.Show()
    app.MainLoop()
