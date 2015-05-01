import argparse
from collections import defaultdict
import os
import pandas as pd
import threshold_node
import matplotlib.pyplot as plt
import glob
import json
import analysis_utils

from sys import stdout


def get_info(directory):
    info = {}
    for i in glob.glob(directory + "/*.json"):
        with open(i) as f:
            vals = json.load(f)
            for v in vals:
                info[v] = vals[v]
    return info


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


def main_experiment(params_df, info, actions, no_actions, advance_scores, no_adv_scores):
    print '{:25s}{:25s}{:10s}{:10s}{:10s}'.format("Source", "File", "lineno", "Comparisons", "Flops")
    for key, group in params_df.groupby('key'):
        f = info[key]['file'][info[key]['file'].rindex('/')+1:]
        print '{:25s}{:25s}{:10d}{:10d}{:10d}'.format(
            info[key]['source'],
            f,
            info[key]['lineno'],
            len(group),
            len(group[group['flop']]),
        )

    for i in actions:
        for j in actions[i]:
            scores = analysis_utils.get_advance_scores(j, params_df, info)
            scores = [(k, v) for k, v in scores.iteritems()]
            scores = sorted(scores, key=lambda x: x[1])
            scores = [(info[x[0]]['source'], x[1], x[0]) for x in scores]
            print len(scores)
            print 'User: ', i
            print '{:20s}{:10s}'.format('Source', 'Score')
            for score in scores:
                print '{:20s}{:10.3f}{:10.3f}'.format(score[0], score[1], advance_scores.loc[advance_scores.index.asof(j),score[2]])

    for i in no_actions:
        for j in no_actions[i]:
            print j
            scores = analysis_utils.get_no_advance_scores(j, params_df, info)
            scores = [(k, v) for k, v in scores.iteritems()]
            scores = sorted(scores, key=lambda x: x[1])
            scores = [(info[x[0]]['source'], x[1], x[0]) for x in scores]




def produce_score_array(params_df, info, fbase):
    adv_data_dict = defaultdict(list)
    #enumerate through all of the indexs and get the scores.
    for count, idx in enumerate(params_df.index):
        scores = analysis_utils.get_advance_scores(idx, params_df, info)
        for s, v in scores.iteritems():
            adv_data_dict[s].append(v)
        stdout.write("\rDoing advance Scores: {:.1%}".format(float(count) / len(params_df)))
    adv_scores = pd.DataFrame(data=adv_data_dict, index=params_df.index)
    adv_scores.to_csv(fbase + '_advance_scores.csv')

def produce_score_array_sampled(params_df, info, fbase):
    a = params_df.index
    time_index = pd.date_range(a[0], a[-1], freq='100L')
    print len(time_index)
    adv_data_dict = defaultdict(list)
    #enumerate through all of the indexs and get the scores.
    for count, idx in enumerate(time_index):
        scores = analysis_utils.get_advance_scores(idx, params_df, info)
        for s, v in scores.iteritems():
            adv_data_dict[s].append(v)
        stdout.write("\rDoing advance Scores: {:.1%}".format(float(count) / len(time_index)))
    adv_scores = pd.DataFrame(data=adv_data_dict, index=time_index)
    adv_scores.to_csv(fbase + '_advance_scores.csv')


def main():
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument('-b', '--bag_record', required=True)
    parser.add_argument('-d', '--info_directory', required=True)
    parser.add_argument('--create_scores', help='Create score csv files', action='store_true')
    parser.add_argument('--namespace', )
    parser.add_argument('--namespace_thresh', )
    args = parser.parse_args()

    node = threshold_node.ThresholdNode(False)
    node.import_bag_file(args.bag_record, args.namespace, args.namespace_thresh)
    thresh_df = node.get_new_threshold_data()
    info = get_info(args.info_directory)
    param_keys = analysis_utils.get_param_keys(info)
    params_only = thresh_df[thresh_df['key'].apply(lambda param: param in param_keys)]
    no_params = thresh_df[thresh_df['key'].apply(lambda no_param: no_param not in param_keys)]
    actions, no_actions = get_marks(args.bag_record)
    b, _ = os.path.splitext(args.bag_record)
    if args.create_scores:
        produce_score_array_sampled(params_only, info, b)
    else:
        advance_scores = pd.read_csv(b + '_advance_scores.csv', index_col=0, parse_dates=True )
        main_experiment(params_only, info, actions, no_actions, advance_scores, pd.DataFrame())


if __name__ == '__main__':
    main()
