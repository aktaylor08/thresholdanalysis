import glob
import argparse

import yaml
import pandas as pd
import matplotlib.pyplot as plt

from thresholdanalysis.analysis import analysis_utils
from thresholdanalysis import config
from thresholdanalysis.analysis.analysis_utils import index_to_float, time_to_float


def main():
    directory = '/Users/ataylor/Reserch/thesis_data/nav_experiment/'
    if directory[-1] != '/':
        directory += '/'
    thresh_dir = directory + 'thresh_dfs/' #args.thresh_dir
    csv_dir = directory + 'csvs/' #args.csv_dir
    info_dir = directory + 'static_info/'
    mapping = {}
    with open(directory + 'mapping.yml') as f:
        mapping = yaml.safe_load(f)
    output = 'navigation'
    if thresh_dir[-1] != '/':
        thresh_dir += '/'
    if info_dir[-1] != '/':
        info_dir += '/'
    if csv_dir[-1] != '/':
        csv_dir += '/'
    for f in glob.glob(csv_dir + "*.csv"):
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
        print f
        print len(act)
        print len(noact)


        # Graph!
        fig, axes = plt.subplots(2, 1, sharex=True)
        x = df['amcl_pose__pose_pose_position_x'].dropna()
        z = df['amcl_pose__pose_pose_position_y'].dropna()
        idx = index_to_float(x.index, first)
        axes[0].plot(idx,  x, linewidth=3)
        axes[0].set_ylabel("X Position (m)", fontsize=15)
        axes[1].plot(idx, z, linewidth=3)
        axes[1].set_xlabel("Elapsed Time (s)", fontsize=15)
        axes[1].set_ylabel("Y Position (m)", fontsize=15)
        axes[0].set_xlim(left=0, right=idx[-1])
        analysis_utils.add_user_marks(act, noact, fig=fig, ax=axes[0], vals=[1.0, 1.1])
        analysis_utils.add_user_marks(act, noact, fig=fig, ax=axes[1])

        a = analysis_utils.key_from_file(f,mapping)
        name = a['name'].replace(' ', '_')
        plt.savefig(config.FIGURE_DIR + output + '_' + name + ".png")


if __name__ == '__main__':
    main()