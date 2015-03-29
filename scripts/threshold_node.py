try:
    import rospy
    from std_msgs.msg import String, Time
    ros = True
except ImportError:
    print("No ros must import to the node instead of live")
    ros = False

try:
    import rosbag_pandas
    rbp = True
except ImportError:
    print ("NO rosbag pandas can only import csv bags")
    rbp = False

import pandas as pd
import os
from threading import Lock


class ThresholdNode(object):
    """Node that controls all threshold stuff"""

    def __init__(self, live):
        print live
        if live:
            if not ros:
                print "NO ROS"
            else:
                print 'hi'
                rospy.init_node('threshold_monitor_node')
                rospy.Subscriber('threshold_information', String, self.thresh_callback)
                rospy.Subscriber('mark_action', Time, self.mark_action_callback)
                rospy.Subscriber('mark_no_action', Time , self.mark_no_action_callback)

        self._marks = []
        self._new_marks = []
        self._data = pd.DataFrame()
        self._new_data = []
        self._lock = Lock()

        self.new_data_store = {}
        self.indexes = {}
        self.times = set()


    # ROS CALLBACKS
    def mark_action_callback(self, msg):
        self.handle_mark(msg.data.sec, msg.data.nsec, True)

    def mark_no_action_callback(self, msg):
        self.handle_mark(msg.data.sec, msg.data.nsec, True)

    def thresh_callback(self, msg):
        self.handle_thresh_string(msg.data)

    # Handle a mark
    def handle_mark(self, sec, n_sec, action):
        """Create a new mark when one is received either through the bag or through
        the user"""
        time = sec + n_sec / 1000000000.0
        time = pd.to_datetime(time, unit='s')
        self._lock.acquire()
        self._new_marks.append((time, action))
        self._marks.append((time, action))
        self._lock.release()

    def get_new_marks(self):
        """Get a new mark from the user when needed"""
        self._lock.acquire()
        ret = [x for x in self._new_marks]
        self._new_marks = []
        self._lock.release()
        return ret

    def get_marks(self):
        """Get all of the recorded marks"""
        self._lock.acquire()
        ret = self._marks[:]
        self._lock.release()
        return ret

    # Handle a new threshold
    def handle_thresh_string(self, thresh_string):
        """Parse a threshold string from an isntrumentation site"""
        vals = thresh_string.split(',')

        # is it an old or new style
        old = False
        if len(vals) % 2 == 1:
            old = True

        # parse times
        time = pd.to_datetime(float(vals[0]), unit='s')
        self.times.add(time)

        # get key result and rest
        if old:
            file_name = vals[1]
            lineno = vals[2]
            thresh_key = file_name + ':' + lineno
            try:
                result = int(vals[3]) == 0
            except:
                result = vals[3] == 'True'
            rest = vals[4:]
        else:
            thresh_key = vals[1]
            try:
                result = int(vals[2]) == 0
            except:
                result = vals[2] == 'True'
            rest = vals[3:]

        # add to data store key and result
        self.add_to_data('key', time, thresh_key, )
        self.add_to_data('result', time, result, )

        # now get the rest of the values here and add them to the data store
        for values in rest:
            try:
                key, val = values.split(':')
                self.add_to_data(key, time, float(val), )
            except ValueError:
                pass

    def add_to_data(self, key, idx_time, value):
        """Add information to the datastore"""
        if key in self.new_data_store:
            self.new_data_store[key].append(value)
        else:
            self.new_data_store[key] = [value]
        if key in self.indexes:
            self.indexes[key].append(idx_time)
        else:
            self.indexes[key] = [idx_time]

    def get_new_threshold_data(self):
        """Get all new threshold information accumulated"""
        # create new threshold and do stuff with it
        self._lock.acquire()
        dataframe = pd.DataFrame(index=list(self.times))
        for dkey in self.new_data_store.iterkeys():
            s = pd.Series(data=self.new_data_store[dkey], index=self.indexes[dkey])
            dataframe[dkey] = s.groupby(s.index).first().reindex(dataframe.index)

        # clear new data
        self.new_data_store.clear()
        self.times.clear()
        self.indexes.clear()

        self._lock.release()
        # sort and return
        dataframe.sort_index(inplace=True)
        return dataframe

    def get_threshold_data(self):
        """Get all threshold information"""
        return pd.DataFrame.copy(self._data)

    # IMPORT
    def import_bag_file(self, bag_file, ns=None):
        if not os.path.exists(bag_file):
            print "Error bag file doesn't exist"
            return
        _, ext = os.path.splitext(bag_file)
        bag_df = None
        if ext == '.bag':
            if rbp:
                bag_df = rosbag_pandas.bag_to_dataframe(bag_file)
            else:
                print "No rosbag pandas cannot read bag!"

        elif ext == '.csv':
            bag_df = pd.read_csv(bag_file, parse_dates=True, index_col=0)
        if ns is None:
            thresh = bag_df['threshold_information__data'].dropna()
        else:
            thresh = bag_df[ns + '_threshold_information__data'].dropna()
        for i in thresh.values:
            self.handle_thresh_string(i)
        self.handle_mark_df(bag_df)




    def import_thresh_file(self, thresh_file):
        """Import a previously parsed threshold file"""
        df = pd.read_csv(thresh_file, parse_dates=True, index_col=0)
        columns = df.columns
        for time, values in zip(df.index, df.values):
            self.times.add(time)
            for key, val in zip(columns, values):
                if key != 'file_name' and key != 'line_number':
                    self.add_to_data(key, time, val)

    def import_mark_file(self, fname):
        """import the mark file"""
        if not os.path.exists(fname):
            print "Error mark file doesn't exist"
            return
        _, ext = os.path.splitext(fname)

        # import the file
        bag_df = None
        if ext == '.bag':
            if rbp:
                bag_df = rosbag_pandas.bag_to_dataframe(fname,
                                                        include=['mark_no_action__data_secs',
                                                                 'mark_no_action__data_nsecs',
                                                                 'mark_action__data_secs', 'mark_action__data_nsecs'])
            else:
                print "No rosbag pandas cannot read bag!"

        elif ext == '.csv':
            bag_df = pd.read_csv(fname, parse_dates=True, index_col=0)

        else:
            print "Unknown File type ", fname
        self.handle_mark_df(bag_df)

    def handle_mark_df(self, bag_df):
        # handle all of the marks
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
                    self.handle_mark(s, ns, True)
