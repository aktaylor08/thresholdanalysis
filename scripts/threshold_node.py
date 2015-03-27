try:
    # import rospy
    # from std_msgs.msg import String
    # import rosbag_pandas
    pass
except ImportError:
    print("No ros must import to the node instead of live")
    ros = False

import pandas as pd
import numpy as np
import os
from threading import Lock


class ThresholdNode(object):

    def __init__(self, live):
        print 'hey there'
        if live:
            if not ros:
                print "NO ROS"
            else:
                pass
            
        self._dirty = False
        self._df = pd.DataFrame()
        self._marks = []
        self._new_marks = []
        self._new_data = []
        self._lock = Lock()


    # ROS CALLBACKS
    def mark_adv_callback(self, msg):
        pass

    def mark_no_adv_callback(self, msg):
        pass

    def thresh_callback(self, msg):
        pass

    # Handle a mark
    def handle_mark(self, n_sec, sec, action):
        time = sec + n_sec / 1000000000.0
        time = pd.to_datetime(time, unit='s')
        self._lock.acquire()
        self._new_marks.append((time, action))
        self._lock.release()

    # Handle a new threshold
    def handle_thresh_string(self, thresh_string):
        pass

    # GETTERS
    def get_data_frame(self):
        if self._dirty:
            # handle dirty data frame
            pass
        return self._df

    def get_marks(self):
        pass

    def get_new_marks(self):
        pass

    # IMPORT
    def import_bag_file(self, msg):
        pass

    def import_thresh_file(self, msg):
        pass

    def import_mark_file(self, fname):
        '''import the mark file'''

        if not os.path.exists(fname):
            print "Error file doesn't exist"
            return
        _, ext = os.path.splitext(fname) 

        bag_df = None
        if ext == '.bag':
            if not ros:
                print 'No ROS on bagfile'
            else:
                # do rosbag info here
                pass
        elif ext == '.csv':
            bag_df = pd.read_csv(fname, parse_dates=True, index_col=0)
        else:
            print "I do not no what to do"

        if bag_df is not None:
            if 'mark_no_action__data_nsecs' in bag_df.columns:
                idx = bag_df.mark_no_action__data_nsecs.dropna().index
                vals = bag_df.loc[idx, ['mark_no_action__data_secs', 'mark_no_action__data_nsecs']]
                for _, data in vals.iterrows():
                    s = data['mark_no_action__data_secs']
                    ns = data['mark_no_action__data_nsecs']
                    self.handle_mark(s, ns, False)
            if 'mark_action__data_nsecs' in bag_df.columns:
                idx = bag_df.mark_action__data_nsecs.dropna().index
                vals = bag_df.loc[idx, ['mark_action__data_secs', 'mark_action__data_nsecs']]
                for _, data in vals.iterrows():
                    s = data['mark_action__data_secs']
                    ns = data['mark_action__data_nsecs']
                    self.handle_mark(s, ns, False)
