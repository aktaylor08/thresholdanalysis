import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def get_param_keys(static_info):
    params = {x: y for x, y in static_info.iteritems() if y['type'] == 'Parameter'}
    return [x['key'] for x in params.itervalues()]

def get_flops(thresh_df, time, static_info, flop_window=5.0, ):
    flop_counts = {}
    param_keys = get_param_keys(static_info)
    if flop_window is not None:
        st = time - datetime.timedelta(seconds=flop_window)
    else:
        st = thresh_df.index[0]
    maxf, minf = 0, 99999
    # get a count for the number of flops in the last n seconds
    for key, data in thresh_df.groupby('key'):
        if key not in param_keys:
            continue
        data = data.between_time(st,time)
        flops = len(data[data['flop']])
        flop_counts[key] = flops
        if flops > maxf:
            maxf = flops
        if flops < minf:
            minf = flops
    return flop_counts, maxf, minf


def get_advance_scores(time, thresh_df, static_info, flop_window=5.0, alpha=1.0, beta=0.6, gamma=0.0):
    param_keys = get_param_keys(static_info)
    # count the flops in the past n seconds.
    flop_counts, minf, maxf = get_flops(thresh_df, time, static_info, None)
    scores = {}

    # now calculate the scores.
    for key, data in thresh_df.groupby('key'):
        if key not in param_keys:
            continue
        try:
            index = data.index.asof(time)
            lf = data.loc[index, 'last_cmp_flop']
            tdelta = (time - lf).total_seconds()
        except TypeError:
            tdelta = 9999.9

        s1 = np.power(tdelta, alpha)
        fc = flop_counts[key] + 1
        s2 = np.power(fc, beta)
        s3 = np.power(static_info[key]['distance'], gamma)
        scores[key] = s1 * s2 * s3
    return scores


def get_advance_results(mark,  thresh_df, static_info, time_limit=3.0,):
        param_keys = get_param_keys(static_info)
        results = []
        # compute time limits and the like
        time = mark.time
        st = time - datetime.timedelta(seconds=time_limit)
        et = time + datetime.timedelta(seconds=time_limit)

        # get data
        if len(thresh_df) == 0:
            return results
        thresh_df = thresh_df.between_time(st, et)
        calc_data = thresh_df.between_time(st, time)
        calc_groups = calc_data.groupby('key')
        elapsed_times = {}
        for key, data in calc_groups:
            if key not in param_keys:
                continue
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
            try:
                elapsed = elapsed_times[key]
            except:
                "error not in there", key
                elapsed = 9999
            if elapsed < time_limit:
                score = elapsed
                sugestion = get_suggestion(time, data, 'cmp', 'thresh', 'res', True)
                # build up the result
                one_result = AnalysisResult()
                thresh_information = static_info[key]

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
        return results


def get_no_advance_scores(time, thresh_df, static_info, flop_window=12.0, alpha=1.0, beta=1.0, gamma=0.0, delta=0.0):
        param_keys = get_param_keys(static_info)
        flop_counts, minf, maxf = get_flops(thresh_df, time, static_info, flop_window)
        results = {}
        if len(thresh_df) == 0:
            print 'empty data set'
            return results
        # get max and min values
        maxes = thresh_df.groupby('key').max()
        mins = thresh_df.groupby('key').min()
        # loop through all of the data points.
        for key, data in thresh_df.groupby('key'):
            if key not in param_keys:
                continue
            calc_data = data.between_time(time - datetime.timedelta(seconds=flop_window), time)
            thresh_information = static_info[key]
            maxval = max(maxes.loc[key, 'thresh'], maxes.loc[key, 'cmp'])
            minval = min(mins.loc[key, 'thresh'], mins.loc[key, 'cmp'])
            cseries = calc_data['cmp']
            const = calc_data['thresh']
            score = 9999.9
            if np.isnan(maxval) or np.isnan(minval):
                score = 9999.9
            elif maxval - minval == 0:
                score = 9999.9
            else:
                cseries = (cseries - minval) / (maxval - minval)
                const = (const - minval) / (maxval - minval)
                if len(cseries) == 0:
                    score = 9999.9
                else:
                    dist = cseries - const
                    dist = np.sqrt(dist * dist).mean()

                    same_count = len(calc_data[((calc_data['cmp'] - calc_data['thresh']).apply(np.abs)) > .0001])
                    # TODO Take into account other thresholds?
                    # calculate the number of different values on each of the other thresholds.
                    num_comparisions = float(thresh_information['other_thresholds'])
                    if num_comparisions == 0:
                        num_comparisions = 1

                    distance = thresh_information['distance']
                    maxdistance = get_max_distance(static_info)
                    s1 = np.power(dist, alpha)
                    fc = 1
                    if flop_counts[key] == 0:
                        fc = 1
                    else:
                        fc = flop_counts[key]
                    s2 = np.power(fc, beta)
                    if same_count == 0:
                        s3 = 1
                    else:
                        s3 = np.power(same_count, gamma)
                    s4 = np.power(distance / maxdistance, delta)
                    score = s1 * s2 * s3 * s4
            results[key] = score
        return results


def get_no_advance_results(mark, thresh_df, static_info, time_limit=5.0, graph_limit=10.0):
        param_keys = get_param_keys(static_info)
        results = []
        if len(thresh_df) == 0:
            print 'empty data set'
            return results
        # compute time limits and the like
        time = mark.time
        # get data
        maxes = thresh_df.groupby('key').max()
        mins = thresh_df.groupby('key').min()
        for key, data in thresh_df.groupby('key'):
            if key not in param_keys:
                continue
            calc_data = data.between_time(time - datetime.timedelta(seconds=time_limit), time)
            thresh_information = static_info[key]
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
            md = get_max_distance(static_info)
            num_comparisions = float(thresh_information['other_thresholds'])
            if num_comparisions == 0:
                num_comparisions = 1
            d_score = d / float(md)
            if d_score == 0:
                d_score = 1
            score = (dist / d_score) + flop_in_series  # / num_comparisions)

            suggestion = get_suggestion(time, calc_data, 'cmp',
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


def get_suggestion(time, data, comp_key, thresh_key, res_key, action, time_limit=5.0):
        """Get a suggestion based on other values"""
        # TODO this still needs some work..
        if action:
            lf = get_series_flops(
                data.between_time(time - datetime.timedelta(seconds=time_limit),
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


def get_series_flops(series):
    """Get the flops in a series of values
    Will return times of every time the series changes from True to False"""
    a = series.dropna()
    fvals = a[a != a.shift(1)]
    fvals = fvals.index[1:]
    return fvals


def get_max_distance(info):
        max_val = 0
        for i in info.itervalues():
            if int(i['distance']) > 0:
                max_val = int(i['distance'])
        return max_val


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


class GraphStorage(object):
    def __init__(self):
        self.index = []
        self.cmp = []
        self.thresh = []
        self.name = ''
        self.suggestion = ''
