import glob
import numpy as np
import argparse

import yaml
import pandas as pd
import matplotlib.pyplot as plt

from thresholdanalysis.analysis import analysis_utils
from thresholdanalysis import config
from thresholdanalysis.analysis.analysis_utils import index_to_float, time_to_float, get_info, get_threshold_dfs, \
    get_threshdf_from_other_file


def main():
    directory = '/Users/ataylor/Research/thesis_data/water_experiment/'
    if directory[-1] != '/':
        directory += '/'
    thresh_dir = directory + 'thresh_dfs/' #args.thresh_dir
    csv_dir = directory + 'csvs/' #args.csv_dir
    info_dir = directory + 'static_info/'
    with open(directory + 'mapping.yml') as f:
        mapping = yaml.safe_load(f)
    output = 'water'
    if output is not None:
        print "Outputting with prefix", output
    else:
        output = 'water_'
    if thresh_dir[-1] != '/':
        thresh_dir += '/'
    if info_dir[-1] != '/':
        info_dir += '/'
    if csv_dir[-1] != '/':
        csv_dir += '/'

    satic_info = get_info(info_dir)
    thresh_dfs, big_df = get_threshold_dfs(thresh_dir,satic_info, mapping)
    for f in glob.glob(csv_dir + "*.csv"):
        thresh_df= get_threshdf_from_other_file(f, mapping, thresh_dfs)
        thresh_df.index = pd.to_datetime(thresh_df.index)
        first = pd.to_datetime(thresh_df.index[0])
        last = pd.to_datetime(thresh_df.index[-1])

        df = pd.read_csv(f, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index)
        first = pd.to_datetime(df.index[0])
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
        act = time_to_float(act, first)
        noact = time_to_float(noact, first)


        # Graph!
        fig, axes = plt.subplots(2, 1, sharex=True)
        df = pd.read_csv(f + '_back', parse_dates=True, index_col=0)
        x = df['vicon_DEMO_WATER_DEMO_WATER__transform_translation_x'].dropna().values
        z = df['vicon_DEMO_WATER_DEMO_WATER__transform_translation_z'].dropna().values
        x = np.append(x, [x[-1]])
        z = np.append(z, [z[-1]])
        idx = index_to_float(df['vicon_DEMO_WATER_DEMO_WATER__transform_translation_x'].dropna().index, first)
        idx = np.append(idx, [(last-first).total_seconds()])
        axes[0].plot(idx,  x, linewidth=3)
        axes[0].set_ylabel("X Position (m)", fontsize=15)
        axes[1].plot(idx, z, linewidth=3)
        axes[1].set_xlabel("Elapsed Time (s)", fontsize=15)
        axes[1].set_ylabel("Height (m)", fontsize=15)
        axes[0].set_xlim(left=0, right=idx[-1])
        analysis_utils.add_user_marks(act, noact, fig=fig, ax=axes[0], vals=[1.0, 1.1])
        analysis_utils.add_user_marks(act, noact, fig=fig, ax=axes[1])

        axes[0].set_xlim(0, (last-first).total_seconds())
        axes[1].set_xlim(0, (last -first).total_seconds())

        a = analysis_utils.key_from_file(f,mapping)
        name = a['name'].replace(' ', '_')
        plt.savefig(config.FIGURE_DIR + 'water_' + name + ".png")


if __name__ == '__main__':
    main()