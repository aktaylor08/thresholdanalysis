from collections import defaultdict
import datetime
import glob
import json
from sys import stdout
import numpy as np
import pandas as pd
import csv



import matplotlib.dates as mdates
import matplotlib.cm as colormaps
import matplotlib.pyplot as plt
from thresholdanalysis.runtime import threshold_node


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
    error_count = 0
    total_count = 0

    # now calculate the scores.
    for key, data in thresh_df.groupby('key'):
        if key not in param_keys:
            continue
        try:
            total_count += 1
            index = data.index.asof(time)
            lf = data.loc[index, 'last_cmp_flop']
            tdelta = (time - lf).total_seconds()
        except TypeError:
            error_count += 1
            tdelta = 9999.9
        except ValueError:
            error_count += 1
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


def get_info(directory):
    info = {}
    for i in glob.glob(directory + "/*.json"):
        with open(i) as f:
            vals = json.load(f)
            for v in vals:
                info[v] = vals[v]
    return info


def calculate_ranking(score_df, zero_bad=False):
    cols = score_df.columns
    num = len(cols)
    col_map = {v: k for k,v in enumerate(cols)}
    idxes = []
    store = np.zeros((len(score_df), len(cols)))
    rc = 0
    for idx, row in score_df.iterrows():
        idxes.append(idx)
        row.sort()
        count = 0
        for name, val in row.iteritems():
            col = col_map[name]
            if zero_bad:
                if val < 9999:
                    store[rc, col] = 1 - ((float(count)) / num)
                    count += 1
            else:
                store[rc, col] = 1 - ((float(count)) / num)
                count += 1
        rc += 1
    return pd.DataFrame(data=store, index=idxes, columns=cols)


def get_df(bag_f, info, parsed=True):
    if not parsed:
        print 'Loading file {:s}'.format(bag_f)
        node = threshold_node.ThresholdNode(False)
        node.import_bag_file(bag_f,)
        thresh_df = node.get_new_threshold_data()
    else:
        thresh_df = pd.read_csv(bag_f, index_col=0, parse_dates=True)
    param_keys = get_param_keys(info)
    params_only = thresh_df[thresh_df['key'].apply(lambda param: param in param_keys)]
    return params_only

def to_latex(df, fout):
        idx = [x.replace('_', '\_') for x in df.index]
        df.index = idx
        df.to_csv(fout, sep='&', line_terminator='\\\\\n',
                        float_format='%.1f', quoting=csv.QUOTE_NONE,
                        escapechar=' ', index_label='Paramter')



def get_marks(bagfilename, number=3):
    df = pd.read_csv(bagfilename, index_col=0, parse_dates=True, quotechar='"')
    actions = {}
    no_actions = {}
    action = 'mark_action'
    no_action = 'mark_no_action'
    othersa = ['marks{:d}_mark_action'.format(i) for i in range(number)]
    othersn = ['marks{:d}_mark_no_action'.format(i) for i in range(number)]
    actions['ctrl'] = get_times(df, action)
    no_actions['ctrl'] = get_times(df, no_action)
    for idx, i in enumerate(othersa):
        actions[idx] = get_times(df, i)
    for idx, i in enumerate(othersn):
        no_actions[idx] = get_times(df, i)
    return actions, no_actions


def get_times(df, start_str):
    nsecs = start_str + '__data_nsecs'
    secs = start_str + '__data_secs'
    ret_val = []
    if nsecs not in df.columns:
        return ret_val
    else:
        idx = df[secs].dropna().index
        vals = df.loc[idx, [secs, nsecs]]
        for _, data in vals.iterrows():
            s = data[secs]
            ns = data[nsecs]
            time = s + ns / 1000000000.0
            time = pd.to_datetime(time, unit='s')
            ret_val.append(time)
    return ret_val

def produce_score_array_sampled(params_df, info, ):
    a = params_df.index
    time_index = pd.date_range(a[0], a[-1], freq='100L')
    adv_data_dict = defaultdict(list)
    no_adv_data_dict = defaultdict(list)
    #enumerate through all of the indexs and get the scores.
    for count, idx in enumerate(time_index):
        scores = get_advance_scores(idx, params_df, info)
        noscores = get_no_advance_scores(idx, params_df, info)
        for s, v in scores.iteritems():
            adv_data_dict[s].append(v)
        for s, v in noscores.iteritems():
            no_adv_data_dict[s].append(v)
        stdout.write("\rDoing Scores: {:.1%}".format(float(count) / len(time_index)))
    adv_scores = pd.DataFrame(data=adv_data_dict, index=time_index)
    no_adv_scores = pd.DataFrame(data=no_adv_data_dict, index=time_index)
    return adv_scores, no_adv_scores



def get_score_dfs(bag_f, info):
    print 'Getting score arrays for {:s}'.format(bag_f)
    thresh_df = get_df(bag_f, info)
    return produce_score_array_sampled(thresh_df, info)


def calculate_ranking(score_df, zero_bad=False):
    cols = score_df.columns
    num = len(cols)
    col_map = {v: k for k,v in enumerate(cols)}
    idxes = []
    store = np.zeros((len(score_df), len(cols)))
    print store.shape
    rc = 0
    for idx, row in score_df.iterrows():
        idxes.append(idx)
        row.sort()
        count = 0
        for name, val in row.iteritems():
            col = col_map[name]
            if zero_bad:
                if val < 9999:
                    store[rc, col] = 1 - ((float(count)) / num)
                    count += 1
            else:
                store[rc, col] = 1 - ((float(count)) / num)
                count += 1
        rc += 1
    return pd.DataFrame(data=store, index=idxes, columns=cols)

def fix_and_plot_color(score, collapse, mod_key, fig=None, ax=None, width=2):
    # get rankings
    ranking = calculate_ranking(score, collapse)
    # Gather zero

    zero_row = (ranking == 0).any(axis=1)
    zero_series = zero_row.apply(lambda x: 0 if x else np.NaN)

    # get the modkey zero row
    if mod_key in ranking.columns:
        zero_mod = ranking[mod_key].copy()
        zero_mod[zero_mod != 0] = np.NaN
    else:
        print "{:s} not in ranking ranks".format(mod_key)
        zero_mod = None

    have_vals = ranking != 0
    have_vals = have_vals.any()[have_vals.any()]
    have_vals = have_vals.index
    # get rid of crap that doesn't exist
    ranking = ranking[have_vals]

    rank_change = ranking != ranking.shift(1)
    ac = rank_change.any(axis=1)
    rc_index = ac[ac]
    rc_index = rc_index[1:]
    indexes = [ranking.index.get_loc(x) for x in rc_index.index]
    ranks = []
    li = 0
    for i in indexes:
        slice = ranking.iloc[li:i+1, :].copy()
        ranks.append(slice)
        li = i
    ranks.append(ranking[li:])
    for i in ranks:
        # get errors here...
        i.iloc[-1,:] = i.iloc[0, :]


    # Create color maps
    levels = {}
    cmap = colormaps.get_cmap('terrain')
    for num, val in enumerate(have_vals):
        levels[val] = float(num) / len(have_vals)
    levels = {k: cmap(v) for k, v in levels.iteritems()}
    levels[mod_key] = 'red'

    # ranking[rank_change] = np.NaN
    # get the location of all zeros and replace them with NaN
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(1,1,1)

    for col in ranking.columns:
        if col == mod_key:
            pass
        else:
            for rv in ranks:
                if len(rv) > 0:
                    rv[col].plot(ax=ax, color=levels[col],linewidth=width, aa=False)
    # plot zeros
    zero_series.plot(ax=ax, c='black',zorder=100, linewidth=width, aa=False)
    if zero_mod is not None:
        zero_mod.plot(ax=ax, c='r', zorder=100, linewidth=width, aa=False)
    if mod_key in ranking.columns:
        for rv in ranks:
            if len(rv) > 0:
                rv[mod_key].plot(ax=ax, c='r', linewidth=width, aa=False)

    ax.set_ylabel('Rank Score', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylim(-.05, 1.05)
    return fig, ax


def main():
    s1 = [9999] * 13
    s2 = [9999, 9999, .1, .1, .1, .2, .3 , .3 ,.3 ,.3 ,.3 ,.3, .3]
    s3 = [.5, .5, .5, .5, .5, .5, .5 , .5 ,.5 ,.5 ,.5 , .29, .29]
    s4 = [9999, 9999, .2, .2, .2, .2, .2 , .1 ,.1 ,.1 ,.6 ,.7, .7]
    s1 = s1 * 10
    s2 = s2 * 10
    s3 = s3 * 10
    s4 = s4 * 10
    data = {'s1' : s1, 's2' : s2, 's3' : s3, 's4' : s4}

    idx = [x for x in range(len(s1))]
    df = pd.DataFrame(data=data, index=idx)
    print df
    fig, ax = fix_and_plot_color(df, True, 's2')
    plt.show()


if __name__ == '__main__':
    main()