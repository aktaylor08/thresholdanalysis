from collections import defaultdict
import sys
import glob
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import yaml

from thresholdanalysis.analysis import analysis_utils
from thresholdanalysis.analysis.analysis_utils import get_partial_match
import thresholdanalysis.config as config


def get_name(f, mapping):
    for k in mapping.iterkeys():
        if k in f:
            return mapping[k]['name']

def get_name_text(f, mapping):
    for k in mapping.iterkeys():
        if k in f:
            return mapping[k]['name_text']


def get_scores(t,key, scores):
    row = scores.loc[scores.index.asof(t), :]
    raw = row[key]
    rank = 0
    if raw < 9999:
        sorted_scores = sorted(row.values)
        rank = 1 - (sorted_scores.index(raw) / float(len(row)))
    return raw, rank


def main():
    parser = argparse.ArgumentParser('Create general statistics here')
    parser.add_argument('directory')
    # parser.add_argument('thresh_dir',)
    # parser.add_argument('info_dir',)
    # parser.add_argument('csv_dir',)
    parser.add_argument('--output-prefix')

    args = parser.parse_args()
    directory = args.directory
    if directory[-1] != '/':
        directory += '/'
    thresh_dir = directory + 'thresh_dfs/' #args.thresh_dir
    csv_dir = directory + 'csvs/' #args.csv_dir
    info_dir = directory + 'static_info/'
    mapping = {}
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
    all_dfs = {}
    for f in glob.glob(thresh_dir + '*.csv'):
        df = analysis_utils.get_df(f, info)
        print(len(df))
        all_dfs[get_name(f, mapping)] = df
    print(info)

    # add some information
    for f, df in all_dfs.iteritems():
        df['source'] = df['key'].apply(lambda x: info[x]['source'])
        df['file'] = df['key'].apply(lambda x: info[x]['file'])
        df['lineno'] = df['key'].apply(lambda x: info[x]['lineno'])
        df.to_csv('/Users/ataylor/' + f + 'res.csv')


    # get the total runtime for experiments
    tt = 0
    for f, df in all_dfs.iteritems():
        print f, len(df)
        t = (df.index[-1] - df.index[0]).total_seconds()
        tt += t
        print t
    print 'Total Runtime: ', tt

    # create a big composite data frame
    big_df = pd.DataFrame()
    for df in all_dfs.itervalues():
        big_df = big_df.append(df)

    # how many row comparisions where there
    print "Total Comparisions: ", len(big_df)

    # How many per second?
    print "Comparisions per second: ", float(len(big_df)) / tt

    # flop instances
    print 'flops: ', len(big_df[big_df.flop == True])
    print 'flop %', len(big_df[big_df.flop == True]) / float(len(big_df))

    # Print unique predicate on comparisions
    print "Unique locations compared", len(big_df['key'].unique())

    print "Unique \"Sources\" compared", len(big_df['source'].unique())

    print "Again unique sources", big_df.source.unique()

    keys = []
    size = []
    per_sec = []
    locations = []
    flops = []
    seconds = []
    sec_perc = []
    tp = []
    fp = []
    for src, data in big_df.groupby('source'):
        keys.append(src)
        size.append(len(data))
        per_sec.append(len(data) / tt)
        locations.append(len(data['key'].unique()))
        flops.append(len(data[data['flop'] == True]))
        s = set()
        for i in data.index:
            s.add(str(i.minute) + str(i.second))
        seconds.append(len(s))
        sec_perc.append(int((len(s) / float(tt) * 100)))
        trues = len(data[data['res'] == 1])
        falses = len(data) - trues
        tp.append((float(trues) / (trues + falses)) * 100)
        fp.append((float(falses) / (trues + falses)) * 100)
        print src, set(data['key'].values)

    stats_df = pd.DataFrame(index=keys, data={
        'Locations': locations,
        'Comparisions': size,
        'Frequency': per_sec,
        # 'Seconds Present' : seconds,
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
    for col in stats_df.columns:
        stats_df.loc['\\textbf{mean}', col] = means[col]
        stats_df.loc['\\textbf{median}', col] = medians[col]
        stats_df.loc['\\textbf{std}', col] = stds[col]
        stats_df.loc['\\textbf{minimum}', col] = mins[col]
        stats_df.loc['\\textbf{maximum}', col] = maxs[col]
        stats_df.loc['\\textbf{sum}', col] = sums[col]
    print stats_df

    analysis_utils.to_latex(stats_df, config.TABLE_DIR + output + '_gen_results.csv')

    print "Frequencies: "
    print stats_df_clean.Frequency

    fig, ax = plt.subplots()
    ax.plot(stats_df_clean.Frequency, marker='o',)
    x = [x-.5 for x in range(len(stats_df_clean.Frequency))]
    y = stats_df_clean.Frequency.index.values
    plt.xticks(x, y, rotation=60)
    plt.savefig(config.FIGURE_DIR + output + '_freq_graph.png')
    plt.cla()

    rtp = stats_df_clean['Runtime \%'].copy()
    rtp.sort()
    print "Runtime Percentages: "
    print rtp
    ax = rtp.plot(kind='bar', y='Runtime %', figsize=(11, 8.5), fontsize=10, rot=-60)
    ax = ax.get_axes()
    ax.set_ylabel('Runtime %')
    ax.set_xlabel("Parameter")
    tick = ax.get_xticks()
    tick = [x + .5 for x in tick]
    ax.set_xticks(tick)
    plt.tight_layout()
    plt.savefig(config.FIGURE_DIR + output + '_runtime_percentage.png')
    plt.cla()


    # Now do user stuff...
    data = defaultdict(list)
    index = []
    act_marks = {}
    no_act_marks = {}
    starts = {}

    act_dt = {}
    no_act_dt = {}

    other_data = defaultdict(list)

    for f in glob.glob(csv_dir + "*.csv"):
        df = pd.read_csv(f, parse_dates=True, index_col=0)
        start = pd.to_datetime(df.index[0])
        starts[f] = start
        actions, no_actions = analysis_utils.get_marks(f, 4)
        act = []
        noact = []
        # compile the marks
        for k, x in actions.iteritems():
            if isinstance(k, int):
                for y in x:
                    act.append(y)
                other_data[str(k) + '_act'].append(len(x))
        for k,x in no_actions.iteritems():
            if isinstance(k,int):
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
        print 'No advance marks',  len(noact)
        print 'Advance Marks', len(act)
    mark_df = pd.DataFrame(data=data, index=index)
    mark_df = mark_df.sort_index()

    other_df = pd.DataFrame(data=other_data, index=index)
    other_df = other_df.sort_index()
    sums = other_df.sum()
    medians = other_df.median()
    means = other_df.mean()
    stds = other_df.std()
    mins = other_df.min()
    maxs = other_df.max()
    for col in other_df.columns:
        other_df.loc['\\textbf{mean}', col] = means[col]
        other_df.loc['\\textbf{median}', col] = medians[col]
        other_df.loc['\\textbf{std}', col] = stds[col]
        other_df.loc['\\textbf{minimum}', col] = mins[col]
        other_df.loc['\\textbf{maximum}', col] = maxs[col]
        other_df.loc['\\textbf{sum}', col] = sums[col]
    other_df.to_csv('/Users/ataylor/mark_data.csv')


    out_mark = mark_df.copy()
    out_mark['Total Marks'] = out_mark['Advance Marks'] + out_mark['No Advance Marks']
    out_mark.columns = ['Type I Marks', 'Type II Marks', 'Total Marks']
    sums = out_mark.sum()
    medians = out_mark.median()
    means = out_mark.mean()
    stds = out_mark.std()
    mins = out_mark.min()
    maxs = out_mark.max()
    for col in out_mark.columns:
        out_mark.loc['\\textbf{mean}', col] = means[col]
        out_mark.loc['\\textbf{median}', col] = medians[col]
        out_mark.loc['\\textbf{std}', col] = stds[col]
        out_mark.loc['\\textbf{minimum}', col] = mins[col]
        out_mark.loc['\\textbf{maximum}', col] = maxs[col]
        out_mark.loc['\\textbf{sum}', col] = sums[col]
    analysis_utils.to_latex(out_mark, config.TABLE_DIR + output + '_marks.csv')


    # Make some wicked graphs.
    if not os.path.exists(thresh_dir + '/advance/'):
        print 'Create the score arrays first'
        sys.exit(0)
    if not os.path.exists(thresh_dir + '/no_advance/'):
        print 'Create the score arrays first'
        sys.exit(0)

    adv_scores = {}
    no_adv_scores = {}
    for f in glob.glob(thresh_dir + "/advance/*.csv"):
        adv_scores[f] = pd.read_csv(f,parse_dates=True, index_col=0)

    for f in glob.glob(thresh_dir + "/no_advance/*.csv"):
        no_adv_scores[f] = pd.read_csv(f,parse_dates=True, index_col=0)


    ranking_data = defaultdict(list)
    score_idx = []
    # now build the images.
    for x in mapping:
        id_for_thresh = mapping[x]['key']

        advs = get_partial_match(x, adv_scores)
        nadvs = get_partial_match(x, no_adv_scores)
        start = get_partial_match(x, starts)
        adv_times = get_partial_match(x, act_dt)
        no_adv_times = get_partial_match(x, no_act_dt)
        advpoints = get_partial_match(x, act_marks)
        nopoints = get_partial_match(x, no_act_marks)
        print adv_times

        if id_for_thresh is not None and id_for_thresh != 'None':
            t1 = 0
            t2 = 0
            count = 0
            print id_for_thresh, 'Advances'
            for i in adv_times:
                # Get raw score
                raw, rank = get_scores(i,id_for_thresh,advs)
                print raw, rank
                t1 += raw
                t2 += rank
                count += 1
            if count > 0:
                count = float(count)
                ranking_data['Type I Score'].append(t1/count)
                ranking_data['Type I Rank'].append(t2/count)
            else:
                ranking_data['Type I Score'].append(9999)
                ranking_data['Type I Rank'].append(0)

            t1 = 0
            t2 = 0
            count = 0
            print "No Advances"
            for i in no_adv_times:
                raw, rank = get_scores(i,id_for_thresh,nadvs)
                print raw, rank
                t1 += raw
                t2 += rank
                count += 1
            if count > 0:
                count = float(count)
                ranking_data['Type II Score'].append(t1/count)
                ranking_data['Type II Rank'].append(t2/count)
            else:
                ranking_data['Type II Score'].append(9999)
                ranking_data['Type II Rank'].append(0)
            score_idx.append(mapping[x]['name_text'])



        # nadvs.index = analysis_utils.index_to_float(nadvs.index, start)
        # advs.index = analysis_utils.index_to_float(advs.index, start)
        fig,axes = plt.subplots(2,1, sharex=True)
        fig, ax1 = analysis_utils.create_ranking_graph(advs, mapping[x]['key'], advpoints, nopoints, fig=fig,
                                                       ax=axes[0], start_time=start, add_labels=False)
        fig, ax2 = analysis_utils.create_ranking_graph(nadvs, mapping[x]['key'], advpoints, nopoints, fig=fig,
                                                       ax=axes[1], start_time=start, add_labels=False)
        #fig.savefig(config.FIGURE_DIR + output + '_' + mapping[x]['name'].replace(' ','_' ) + 'no_adv_rank_graph.png')
        ax2.set_xlabel("Elapsed Time (s)")
        ax1.set_ylabel('Type I Ranking')
        ax2.set_ylabel('Type II Ranking')

        fig.savefig(config.FIGURE_DIR +output + '_' + mapping[x]['name'].replace(' ', '_') + 'rankng_graphs.png')

    rank_df = pd.DataFrame(index=score_idx, data=ranking_data)
    rank_df.sort_index(inplace=True)
    analysis_utils.to_latex(rank_df, config.TABLE_DIR + output + '_scores.csv')


if __name__ == '__main__':
    main()
