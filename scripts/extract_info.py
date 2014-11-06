#!/usr/bin/env python
# encoding: utf-8

import rosbag_pandas as rbp
import pandas as pd
import numpy as np
import sys


def add_to_store(store, key, idx, value, length):
    if key in store:
        store[key][idx] = value
    else:
        if isinstance(value, int) or isinstance(value, float):
            arr = np.empty(length)
            arr.fill(np.NAN)
        else:
            arr = np.array([None] * length)
        arr[idx] = value
        store[key] = arr


def to_dataframe(iterable):
    length = len(iterable)
    datastore = {}
    index = np.array([None] * length)
    idx = 0
    for i in iterable:
        vals = i.split(',')
        time = vals[0]
        index[idx] = pd.to_datetime(time)

        fname = vals[1]
        add_to_store(datastore, 'file_name', idx, fname, length)
        lineno = vals[2]
        add_to_store(datastore, 'line_number', idx, lineno, length)
        line = vals[3]
        add_to_store(datastore, 'line', idx, line, length)
        truth = vals[4]
        add_to_store(datastore, 'result', idx, truth, length)
        rest = vals[5:]
        for i in rest:
            try:
                key, val = i.split(':')
            except:
                pass
                # print i 
                # vals = i.split(':')
                # key = vals[0]
                # print key
                # val = ''.join(vals[1:])


            add_to_store(datastore, key, idx, val, length)
        idx +=1
    df = pd.DataFrame(data=datastore, index=index)
    return df

def from_file(fname, return_both=False):
    df = rbp.bag_to_dataframe(fname)
    thresh = to_dataframe(df.threshold_information__data.dropna().values)
    if return_both:
        return thresh, df
    else:
        return thresh




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'need file'
        sys.exit()

    f = sys.argv[1]
    df = rbp.bag_to_dataframe(f)
    data = df.threshold_information__data
    to_dataframe(data.dropna().values)



