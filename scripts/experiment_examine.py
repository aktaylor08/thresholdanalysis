import argparse
import pandas as pd
import threshold_node
import matplotlib.pyplot as plt
import glob
import json
import analysis_utils


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

    for idx, val in actions.iteritems():
        print '\t', idx, val
    print 'No Action:'
    for idx, val in no_actions.iteritems():
        print '\t', idx, val
    return actions, no_actions


def get_times(df, start_str):
    nsecs = start_str + '__data_nsecs'
    secs = start_str + '__data_secs'
    ret_val = []
    print nsecs
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument('-b', '--bag_record', required=True)
    parser.add_argument('-d', '--info_directory', required=True)
    parser.add_argument('--namespace', )
    parser.add_argument('--namespace_thresh', )
    args = parser.parse_args()

    live = False
    node = threshold_node.ThresholdNode(False)
    node.import_bag_file(args.bag_record, args.namespace, args.namespace_thresh)
    thresh_df = node.get_new_threshold_data()
    info = get_info(args.info_directory)
    param_keys = analysis_utils.get_param_keys(info)
    params_only = thresh_df[thresh_df['key'].apply(lambda x: x in param_keys)]
    no_params = thresh_df[thresh_df['key'].apply(lambda x: x not in param_keys)]
    actions, no_actions = get_marks(args.bag_record)

    print '{:25s}{:25s}{:10s}{:10s}{:10s}'.format("Source", "File", "lineno", "Comparisons", "Flops")
    for key, group in params_only.groupby('key'):
        f = info[key]['file'][info[key]['file'].rindex('/')+1 :]
        print '{:25s}{:25s}{:10d}{:10d}{:10d}'.format(
            info[key]['source'],
            f,
            info[key]['lineno'],
            len(group),
            len(group[group['flop']]),
        )

    for i in actions:
        print i, actions[i]
        for j in actions[i]:
            print j
            analysis_utils.get_advanced_scores(j, params_only, info)

