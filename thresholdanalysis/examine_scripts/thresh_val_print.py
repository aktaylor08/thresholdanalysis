import pandas as pd
import argparse
import glob


def main():
    # setup
    parser = argparse.ArgumentParser('Create general statistics here')
    parser.add_argument('directory')
    args = parser.parse_args()
    directory = args.directory
    if directory[-1] != '/':
        directory += '/'
    thresh_dir = directory + 'thresh_dfs/'
    df = pd.DataFrame()
    for f in glob.glob(thresh_dir + '*csv'):
        df = df.append(pd.read_csv(f, index_col=0, parse_dates=True))
    print len(df)
    for key, g in df.groupby('key'):
        print key, g.thresh.unique()

if __name__ == '__main__':
    main()
