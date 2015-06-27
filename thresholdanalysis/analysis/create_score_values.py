import argparse
from collections import defaultdict
import os
import glob
import json
from sys import stdout

import pandas as pd

from thresholdanalysis.runtime import threshold_node
from thresholdanalysis.analysis import analysis_utils


def produce_score_array_sampled(params_df, info, ):
    a = params_df.index
    time_index = pd.date_range(a[0], a[-1], freq='100L')
    print len(time_index)
    adv_data_dict = defaultdict(list)
    no_adv_data_dict = defaultdict(list)
    #enumerate through all of the indexs and get the scores.
    for count, idx in enumerate(time_index):
        scores = analysis_utils.get_advance_scores(idx, params_df, info)
        noscores = analysis_utils.get_no_advance_scores(idx, params_df, info)
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
    thresh_df = get_thresh_df(bag_f)
    param_keys = analysis_utils.get_param_keys(info)
    params_only = thresh_df[thresh_df['key'].apply(lambda param: param in param_keys)]
    return produce_score_array_sampled(params_only, info)



def get_thresh_df(bagf, namespace=None, namespace_thresh=None):
    node = threshold_node.ThresholdNode(False)
    node.import_bag_file(bagf, namespace, namespace_thresh)
    return node.get_new_threshold_data()


def main():
    parser = argparse.ArgumentParser('Show information on the user marks')
    parser.add_argument("directory", help="Directory to parse")
    parser.add_argument('info_directory', help="Info Directory to parse")
    parser.add_argument('--namespace', )
    parser.add_argument('--namespace_thresh', )
    args = parser.parse_args()

    direct = args.directory
    if direct[-1] != '/':
        direct += '/'
    info = analysis_utils.get_info(args.info_directory)
    print os.listdir(direct)
    for f in glob.glob(direct + '*.csv'):
        fbase, _ = os.path.splitext(f)
        fbase = fbase[fbase.rindex('/') + 1 :]
        print direct + 'no_advance/' + fbase + 'no_advance_scores.csv'
        adv, no_adv = get_score_dfs(f, info)
        if not os.path.exists(direct + 'advance/'):
            os.mkdir(direct + 'advance/')
        if not os.path.exists(direct + 'no_advance/'):
            os.mkdir(direct + 'no_advance/')
        adv.to_csv(direct + 'advance/' + fbase + '_advance_scores.csv')
        no_adv.to_csv(direct + 'no_advance/' + fbase + '_no_advance_scores.csv')


if __name__ == '__main__':
    main()
