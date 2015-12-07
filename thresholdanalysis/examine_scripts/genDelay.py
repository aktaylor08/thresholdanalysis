from collections import defaultdict
import json
import glob
import argparse
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import yaml

from thresholdanalysis.analysis import analysis_utils
from thresholdanalysis.analysis.analysis_utils import get_partial_match, get_threshdf_from_other_file, get_threshold_dfs, \
    get_name_text, index_to_float
import thresholdanalysis.config as config


def get_scores(t, key, scores):
    row = scores.loc[scores.index.asof(t), :]
    raw = row[key]
    rank = 0
    if raw < 9999:
        sorted_scores = sorted(row.values)
        rank = 1 - (sorted_scores.index(raw) / float(len(row)))
    return raw, rank


def get_runtime(threshold_data_dfs):
    # get the total runtime for experiments
    tt = 0
    what_what = 0
    print 'length of files :'
    for f, df in threshold_data_dfs.iteritems():
        print '\t', f, len(df)
        what_what += len(df)
        t = (df.index[-1] - df.index[0]).total_seconds()
        tt += t
    print 'Length of all', what_what
    return tt


def print_total_data(big_df, total_time):
    print 'Total Runtime: ', total_time
    # how many row comparisions where there
    print "Total Comparisions: ", len(big_df)
    # How many per second?
    print "Comparisions per second: ", float(len(big_df)) / total_time
    # flop instances
    print 'flops: ', len(big_df[big_df.flop == True])
    print 'flop %', (len(big_df[big_df.flop == True]) / float(len(big_df))) * 100
    # Print unique predicate on comparisions
    print "Unique locations compared", len(big_df['key'].unique())
    print "Unique \"Sources\" compared", len(big_df['source'].unique())
    #print "Again unique sources", big_df.source.unique()
    print 'Unique flopers ', len(big_df[big_df.flop == True]['source'].unique())
    print 'Unique flopers %', len(big_df[big_df.flop == True]['source'].unique()) / float(len(big_df['source'].unique())) * 100



def create_stats_dataframe(big_df, total_time, output):
    keys = []
    counts_of = []
    per_sec = []
    locations = []
    flops = []
    seconds = []
    sec_perc = []
    tp = []
    fp = []
    for src, data in big_df.groupby('source'):
        keys.append(src)
        counts_of.append(len(data))
        per_sec.append(len(data) / total_time)
        locations.append(len(data['key'].unique()))
        flops.append(len(data[data['flop'] == True]))
        s = set()
        for i in data.index:
            s.add(str(i.minute) + str(i.second))
        seconds.append(len(s))
        sec_perc.append(int((len(s) / float(total_time) * 100)))
        trues = len(data[data['res'] == 1])
        falses = len(data) - trues
        tp.append((float(trues) / (trues + falses)) * 100)
        fp.append((float(falses) / (trues + falses)) * 100)

    stats_df = pd.DataFrame(index=keys, data={
        'Locations': locations,
        'Comparisons': counts_of,
        'Frequency': per_sec,
        'Runtime \%': sec_perc,
        'Flops': flops,
        'True \%': tp,
        'False \%': fp,
    },
                            columns=[
                                'Locations',
                                'Comparisons',
                                'Frequency',
                                'Runtime \%',
                                'Flops',
                                'True \%',
                                'False \%',
                            ]
                            )

    sums = stats_df.sum()
    medians = stats_df.median()
    means = stats_df.mean()
    stds = stats_df.std()
    mins = stats_df.min()
    maxs = stats_df.max()
    stats_df_clean = stats_df.copy()
    try:
        print 'freq max', maxs['Frequency']
        print 'freq min', mins['Frequency']
    except:
        pass
    for col in stats_df.columns:
        stats_df.loc['\\textbf{mean}', col] = means[col]
        stats_df.loc['\\textbf{median}', col] = medians[col]
        stats_df.loc['\\textbf{std}', col] = stds[col]
        stats_df.loc['\\textbf{minimum}', col] = mins[col]
        stats_df.loc['\\textbf{maximum}', col] = maxs[col]
        stats_df.loc['\\textbf{sum}', col] = sums[col]
    return stats_df_clean


def plot_runtime_percentages(stats_df_clean, output):
    rtp = stats_df_clean['Runtime \%'].copy()
    rtp.sort()
    ax = rtp.plot(kind='bar', y='Runtime %', figsize=(11, 8.5), fontsize=10, rot=-60)
    ax = ax.get_axes()
    ax.set_ylabel('Runtime %')
    ax.set_xlabel("Parameter")
    tick = ax.get_xticks()
    tick = [x + .5 for x in tick]
    ax.set_xticks(tick)
    plt.tight_layout()
    plt.cla()


def plot_frequencies(stats_df_clean, output):
    fig, ax = plt.subplots()
    ax.plot(stats_df_clean.Frequency, marker='o', )
    x = [x - .5 for x in range(len(stats_df_clean.Frequency))]
    y = stats_df_clean.Frequency.index.values
    plt.xticks(x, y, rotation=60)
    plt.cla()


def main():
    # setup
    parser = argparse.ArgumentParser('Create general statistics here')
    parser.add_argument('directory')
    parser.add_argument('--output-prefix')
    parser.add_argument('--low')
    parser.add_argument('--high')
    args = parser.parse_args()
    directory = args.directory
    if directory[-1] != '/':
        directory += '/'
    thresh_dir = directory + 'thresh_dfs/'
    csv_dir = directory + 'csvs/'
    info_dir = directory + 'static_info/'
    with open(directory + 'mapping.yml') as f:
        mapping = yaml.safe_load(f)
    output = args.output_prefix
    if output is not None:
        print "Outputting with prefix", output
    else:
        output = 'generic_'
    if thresh_dir[-1] != '/':
        thresh_dir += '/'
    if info_dir[-1] != '/':
        info_dir += '/'
    if csv_dir[-1] != '/':
        csv_dir += '/'
    info = analysis_utils.get_info(info_dir)

    # Read all of the data into a dictonary
    threshold_data_dfs,file_map, big_df = get_threshold_dfs(thresh_dir, info, mapping, fmap=True)
    for s, vals in big_df.groupby('source'):
        u = vals['thresh'].unique()
        if len(u) > 1:
            print s, u

    # Now do user stuff...
    data = defaultdict(list)
    index = []
    act_marks = {}
    no_act_marks = {}
    starts = {}

    act_dt = {}
    no_act_dt = {}

    other_data = defaultdict(list)
    all_dem_nos = {}
    all_dem_yes = {}

    for f in glob.glob(csv_dir + "*.csv"):
        tdf = get_threshdf_from_other_file(f, mapping, threshold_data_dfs)
        start = pd.to_datetime(tdf.index[0])
        starts[f] = start
        actions, no_actions = analysis_utils.get_marks(f, 4)
        all_dem_nos[f] =  {}
        all_dem_yes[f] = {}
        act = []
        noact = []
        # compile the marks
        for k, x in actions.iteritems():
            if isinstance(k, int):
                all_dem_yes[f][k] = analysis_utils.time_to_float(x, start)
                for y in x:
                    act.append(y)
                other_data[str(k) + '_act'].append(len(x))
        for k, x in no_actions.iteritems():
            if isinstance(k, int):
                all_dem_nos[f][k] = analysis_utils.time_to_float(x, start)
                for y in x:
                    noact.append(y)
                other_data[str(k) + 'no_act_'].append(len(x))
        # store non translated
        act_dt[f] = act
        no_act_dt[f] = noact
        # Also store translated
        act_marks[f] = analysis_utils.time_to_float(act, start)
        no_act_marks[f] = analysis_utils.time_to_float(noact, start)
        data['Advance Marks'].append(len(act))
        data['No Advance Marks'].append(len(noact))
        index.append(get_name_text(f, mapping))



    # now build the images.
    v = {}
    cool_data = {}
    for x in mapping:
        id_for_thresh = mapping[x]['key']
        df = get_partial_match(x, file_map)
        changed = df[df['key'] == id_for_thresh]
        start = get_partial_match(x, starts)
        if len(changed) > 0:
            times = index_to_float(changed.index, start)
            plt.plot(times, changed['thresh'].values)
            plt.plot(times, changed['cmp'].values)
            plt.plot(times, [mapping[x]['norm'] for _ in times])
            plt.savefig('/Users/ataylor/Desktop/{:s}.png'.format(x))
            sv = {}
            #get and print flops
            abv = changed['cmp'] > changed['thresh']
            sv['real'] = index_to_float(abv[abv != abv.shift()].index, start)[1:].tolist()
            abv = changed['cmp'] > mapping[x]['norm']
            sv['norm'] = index_to_float(abv[abv != abv.shift()].index, start)[1:].tolist()
            v[x] = sv
            if 'err' in mapping[x]:
                a = get_partial_match(x, all_dem_nos)
                b = get_partial_match(x, all_dem_yes)
                marks = {}
                for i in a.keys():
                    val = a[i] + b[i]
                    val.sort()
                    marks[i] = (np.array(val) - mapping[x]['err']).tolist()
                    marks[i] = filter(lambda x: x > 0, marks[i])
                    if len(marks[i]) > 0:
                        marks[i] = min(marks[i])
                    else:
                        marks[i] = np.NaN
                cool_data[mapping[x]['name_text']] = marks





            # Do error calcs for all user marks
        plt.cla()
    with open('/Users/ataylor/Desktop/crap.json', 'w') as fff:
        json.dump(v,fff,indent=2)
    dd = pd.DataFrame(cool_data)
    dd = dd.transpose()
    mm = dd.mean(axis=1)
    print mm
    dd['Mean'] = mm
    analysis_utils.to_latex(dd, config.TABLE_DIR +  output + '_ave_delay.csv', ["User 1", "User 2", "User 3", "User 4", "Mean"])







if __name__ == '__main__':
    main()
