import os
import sys
import yaml
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from thresholdanalysis.analysis import analysis_utils
from thresholdanalysis import config



def main():
    parser = argparse.ArgumentParser('Making water graphs')
    parser.add_argument('directory')
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
        output = 'water_'
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
        fig, axes = plt.subplots(2, 1, sharex=True)
        x = df['vicon_DEMO_WATER_DEMO_WATER__transform_translation_x'].dropna()
        x.index = pd.to_datetime(x.index)
        z = df['vicon_DEMO_WATER_DEMO_WATER__transform_translation_z'].dropna()
        z.index = pd.to_datetime(z.index)
        axes[0].plot(index_to_float(x.index), x, linewidth=3)
        axes[0].set_ylabel("X Position (m)", fontsize=15)
        axes[1].plot(index_to_float(z.index), z, linewidth=3)
        axes[1].set_xlabel("Elapsed Time (s)", fontsize=15)
        axes[1].set_ylabel("Height (m)", fontsize=15)
        axes[0].set_xlim(left=0)
        analysis_utils.add_user_marks(act, noact, fig=fig, ax=axes[0], vals=[1.0, 1.1])
        analysis_utils.add_user_marks(act, noact, fig=fig, ax=axes[1])

        a = analysis_utils.key_from_file(f,mapping)
        name = a['name'].replace(' ', '_')
        plt.savefig(config.FIGURE_DIR + name + ".png")


def index_to_float(idx):
    try:
        idx = pd.to_datetime(idx)
        dates = idx.to_series()
        first = dates.values[0]
        idx = dates.apply(lambda x: (x - first).total_seconds())
        vals = idx.values
        return vals
    except Exception as e:
        print e
        return idx

def time_to_float(actions, first):
    vals = []
    for i in actions:
        vals.append((i - first).total_seconds())
    return vals



if __name__ == '__main__':
    main()