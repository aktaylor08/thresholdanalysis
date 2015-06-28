import argparse
from collections import defaultdict
import os
import glob
import json
from sys import stdout

import pandas as pd

from thresholdanalysis.analysis import analysis_utils





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
        adv, no_adv = analysis_utils.get_score_dfs(f, info)
        if not os.path.exists(direct + 'advance/'):
            os.mkdir(direct + 'advance/')
        if not os.path.exists(direct + 'no_advance/'):
            os.mkdir(direct + 'no_advance/')
        adv.to_csv(direct + 'advance/' + fbase + '_advance_scores.csv')
        no_adv.to_csv(direct + 'no_advance/' + fbase + '_no_advance_scores.csv')


if __name__ == '__main__':
    main()
