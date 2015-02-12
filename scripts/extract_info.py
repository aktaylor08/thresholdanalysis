#!/usr/bin/env python
# encoding: utf-8
import argparse

import os
import warnings
import rosbag_pandas as rbp
import pandas as pd


def add_to_data(key, idx, value, indexes, data_store):
    if key in data_store:
        data_store[key].append(value)
    else:
        data_store[key] = [value]
    if key in indexes:
        indexes[key].append(idx)
    else:
        indexes[key] = [idx]


def to_dataframe(series):
    errors = dict()
    data_dict = {}
    indexes = {}
    times = set()

    for BAG_TIME, i in series.iteritems():
        vals = i.split(',')
        time = pd.to_datetime(float(vals[0]), unit='s')
        times.add(time)

        file_name = vals[1]
        lineno = vals[2]
        thresh_id = file_name + str(lineno)
        thresh_key = file_name + ':' + lineno

        add_to_data('key', time, thresh_key, indexes, data_dict)
        add_to_data('id', time, thresh_id, indexes, data_dict)
        add_to_data('file_name', time, file_name, indexes, data_dict)
        add_to_data('line_number', time, lineno, indexes, data_dict)

        result = bool(vals[3])
        add_to_data('result', time, result, indexes, data_dict)
        rest = vals[4:]
        for values in rest:
            try:
                key, val = values.split(':')
                add_to_data(key, time, val, indexes, data_dict)
            except ValueError:
                errors['Value Error: ' + thresh_key] = errors.get('Value Error: ' + thresh_key, 0) + 1

    dataframe = pd.DataFrame(index=list(times))
    for dkey in data_dict.iterkeys():
        s = pd.Series(data=data_dict[dkey], index=indexes[dkey])
        dataframe[dkey] = s.groupby(s.index).first().reindex(dataframe.index)
    for key, count in errors.iteritems():
        warnings.warn(str(count) + ' ' + str(key))
    return dataframe


def test_time():
    bag = '/home/ataylor/asctec_ws/src/collab_launch/mybags/_2015-01-30-16-53-42.bag'
    name_space = 'a'

    topic_list = ['/mark_no_action', '/mark_action', '/' + name_space + '/threshold_information']
    bads_goods = [u'mark_action__data_nsecs', u'mark_action__data_secs', u'mark_no_action__data_nsecs',
                  u'mark_no_action__data_secs', ]
    file_name, _ = os.path.splitext(bag)
    # df = rbp.bag_to_dataframe(args.bag, include=['/threshold_information'])
    data_frame = rbp.bag_to_dataframe(bag, include=topic_list)

    if name_space == '':
        threshold_data = data_frame['threshold_information__data']
    else:
        threshold_data = data_frame[name_space + '_threshold_information__data']

    thresh_df = to_dataframe(threshold_data.dropna())
    thresh_df.to_csv(file_name + '_thresh.csv', )
    data_frame[bads_goods].to_csv(file_name + '_marked.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Extracting Data', usage='Use to extract threshold information', )
    parser.add_argument('-b', '--bag', help='The bag to extract information from', required=True)
    parser.add_argument('-o', '--output', help='prefix to add to output files')
    parser.add_argument('-n', '--namespace', help='Namespace of the threshold topic')
    args = parser.parse_args()
    if args.namespace is None:
        topics = ['/mark_no_action', '/mark_action', '/threshold_information']
    else:
        topics = ['/mark_no_action', '/mark_action', '/' + args.namespace + '/threshold_information']
    good_bad = [u'mark_action__data_nsecs', u'mark_action__data_secs', u'mark_no_action__data_nsecs',
                u'mark_no_action__data_secs', ]

    if args.namespace is not None:
        ns = args.namespace
    else:
        ns = ''
    fname, _ = os.path.splitext(args.bag)
    if args.output is not None:
        fname = args.output
    df = rbp.bag_to_dataframe(args.bag, include=topics)

    if ns == '':
        data = df['threshold_information__data']
    else:
        data = df[ns + '_threshold_information__data']

    thresh = to_dataframe(data.dropna())
    thresh.sort_index(inplace=True)
    thresh.to_csv(fname + '_thresh.csv', )

    df[[x for x in good_bad if x in df.columns]].to_csv(fname + '_marked.csv')
