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





def get_thresh_df(bagf, namespace=None, namespace_thresh=None):
    node = threshold_node.ThresholdNode(False)
    node.import_bag_file(bagf, namespace, namespace_thresh)
    return node.get_new_threshold_data()


def main():
    info = get_info('/Users/ataylor/Research/thresholdanalysis/test_data/water_sampler/static_info')
    thresh_df = get_thresh_df('/Users/ataylor/Research/thresholdanalysis/test_data/water_sampler/dynamic/all_info/
    param_keys = analysis_utils.get_param_keys(info, )
    params_only = thresh_df[thresh_df['key'].apply(lambda param: param in param_keys)]
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
    adv_scores = pd.DataFrame(data=adv_data_dict, index=time_index)
    no_adv_scores = pd.DataFrame(data=no_adv_data_dict, index=time_index)






if __name__ == '__main__':
    main()
