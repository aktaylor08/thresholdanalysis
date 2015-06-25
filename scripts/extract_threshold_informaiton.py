import glob
import os
import threshold_node
import argparse
import rosbag_pandas

def process_threholds(file_name):
    print("Processing {:s}".format(file_name))
    node = threshold_node.ThresholdNode(False)
    node.import_bag_file(file_name)
    df = node.get_new_threshold_data()
    base,_ = os.path.splitext(file_name)
    fout = base + "_thesholds.csv"
    df.to_csv(fout)

def process_other(file_name, exclude):
    if exclude is None:
        exclude = []
    exclude.append("/threshold_information")
    df = rosbag_pandas.bag_to_dataframe(file_name, exclude=exclude)
    base,_ = os.path.splitext(file_name)
    outf = base + ".csv"
    df.to_csv(outf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--exclude", nargs="*")
    args = parser.parse_args()
    print os.listdir(args.directory)
    for i in glob.glob(args.directory + "/*.bag"):
        process_threholds(i)
        process_other(i, args.exclude)

if __name__ == "__main__":
    main()