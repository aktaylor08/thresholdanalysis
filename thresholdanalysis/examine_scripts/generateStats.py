from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import sys
from thresholdanalysis.analysis import analysis_utils
import glob
import yaml
import os

import argparse

def get_name(f, mapping):
    for k in mapping.iterkeys():
        if k in f:
            return mapping[k]['name']


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
    print mapping
    print thresh_dir
    print csv_dir
    print info_dir
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
        all_dfs[get_name(f, mapping)] = df
    print len(all_dfs)

    # add some information
    for f, df in all_dfs.iteritems():
        df['source'] = df['key'].apply(lambda x: info[x]['source'])
        df['file'] = df['key'].apply(lambda x: info[x]['file'])
        df['lineno'] = df['key'].apply(lambda x: info[x]['lineno'])

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

    # how many comparisions where there
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
                                'Comparisions',
                                'Frequency',
                                # 'Seconds Present',
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
    # print stats_df

    analysis_utils.to_latex(stats_df, '/Users/ataylor/Research/thesis/data/' + output + '_gen_results.csv')

    print "Frequencies: "
    print stats_df_clean.Frequency

    fig, ax = plt.subplots()
    ax.plot(stats_df_clean.Frequency, marker='o',)
    x = [x-.5 for x in range(len(stats_df_clean.Frequency))]
    y = stats_df_clean.Frequency.index.values
    plt.xticks(x, y, rotation=60)
    plt.savefig("/Users/ataylor/Research/thesis/myFigures/" + output + '_freq_graph.png')
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
    plt.savefig("/Users/ataylor/Research/thesis/myFigures/" + output + '_runtime_percentage.png')
    plt.cla()

    # rtp = stats_df_clean['True \%'].copy()
    # rtp.sort()
    # ax = rtp.plot(kind='bar', y='True %', figsize=(11, 8.5), fontsize=10)
    # ax = ax.get_axes()
    # ax.yaxis.set_label('True \%')
    # plt.show()

    # x = stats_df_clean['Flops']
    # y = stats_df_clean['Runtime \%']
    #
    # plt.scatter(x, stats_df_clean.Frequency)
    # plt.show()

    # Now do user stuff...
    mark_frames = {}
    data = defaultdict(list)
    index = []
    act_marks = {}
    no_act_marks = {}

    for f in glob.glob(csv_dir + "*.csv"):
        actions, no_actions = analysis_utils.get_marks(f, 4)
        act = []
        noact = []
        # compile the marks
        for k, x in actions.iteritems():
            if isinstance(k, int):
                for y in x:
                    act.append(y)
        for k,x in no_actions.iteritems():
            if isinstance(k,int):
                for y in x:
                    noact.append(y)
        act_marks[f] = act
        no_act_marks[f] = noact
        data['Advance Marks'].append(len(act))
        data['No Advance Marks'].append(len(noact))
        index.append(get_name(f, mapping))
        print 'No advance marks',  len(noact)
        print 'Advance Marks', len(act)
    mark_df = pd.DataFrame(data=data, index=index)
    mark_df = mark_df.sort_index()
    analysis_utils.to_latex(mark_df, "/Users/ataylor/Research/thesis/data/" + output + '_marks.csv')


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
        print f
        adv_scores[f] = pd.read_csv(f,parse_dates=True, index_col=0)

    for f in glob.glob(thresh_dir + "/no_advance/*.csv"):
        no_adv_scores[f] = pd.read_csv(f,parse_dates=True, index_col=0)
        print f


    # now build the images.
    for x in mapping:
        print x
        advs = get_partial_match(x, adv_scores)
        nadvs = get_partial_match(x, no_adv_scores)

        advpoints = get_partial_match(x, act_marks)
        nopoints = get_partial_match(x, no_act_marks)
        print advpoints, nopoints

        nopoints = zip(nopoints, vals * (len(nopoints) /2))
        advpoints = zip(advpoints, vals * (len(advpoints) /2))

        vals = [.4, .6]








def get_partial_match(key, dicton):
    for k,v in dicton.iteritems():
        if key in k:
            print 'matches', k
            return v


if __name__ == '__main__':
    main()
