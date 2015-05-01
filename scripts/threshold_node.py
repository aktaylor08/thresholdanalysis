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
        if live:
            if not ros:
                print "NO ROS"
            else:
                rospy.init_node('threshold_monitor_node')
                rospy.Subscriber('threshold_information', String, self.thresh_callback)
                rospy.Subscriber('mark_action', Time, self.mark_action_callback)
                rospy.Subscriber('mark_no_action', Time, self.mark_no_action_callback)

        self._marks = []
        self._new_marks = []
        self._data = pd.DataFrame()
        self._new_data = []
        self._lock = Lock()

        self.new_data_store = {}
        self.indexes = {}
        self.last_total_results = {}
        self.last_local_results = {}
        self.last_local_flop = {}
        self.last_total_flop = {}
        self.times = set()

    # ROS CALLBACKS
    def mark_action_callback(self, msg):
        self.handle_mark(msg.data.secs, msg.data.nsecs, True)

    def mark_no_action_callback(self, msg):
        self.handle_mark(msg.data.secs, msg.data.nsecs, False)

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
        """Parse a threshold string"""
        vals = thresh_string.split(',')

        # is it an old or new style
        old = False
        if len(vals) % 2 == 1:
            old = True

        self._lock.acquire()
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
        local_result = None

        # now get the rest of the values here and add them to the data store
        for values in rest:
            try:
                key, val = values.split(':')
                if key == 'res':
                    try:
                        val = int(vals[2]) == 0
                    except:
                        val = vals[2] == 'True'
                    local_result = val
                self.add_to_data(key, time, float(val), )
            except ValueError as e:
                print e

        # keep track of flops here
        # overall on result
        if thresh_key in self.last_total_results:
            if result != self.last_total_results[thresh_key]:
                # it is a flop
                self.last_total_flop[thresh_key] = time
        else:
            self.last_total_flop[thresh_key] = None

        self.last_total_results[thresh_key] = result
        self.add_to_data('last_total_flop', time, None)

        # local results
        local_flop = False
        if thresh_key in self.last_local_results:
            if local_result != self.last_local_results[thresh_key]:
                # it is a flop
                self.last_local_flop[thresh_key] = time
                local_flop = True
        else:
            self.last_local_flop[thresh_key] = None

        self.last_local_results[thresh_key] = local_result
        self.add_to_data('last_cmp_flop', time, self.last_local_flop[thresh_key])
        self.add_to_data('flop', time, local_flop)
        self._lock.release()

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
    def import_bag_file(self, bag_file, ns=None, nst=None):
        if not os.path.exists(bag_file):
            print "Error bag file doesn't exist"
            return
        _, ext = os.path.splitext(bag_file)
        bag_df = None
        csv = False
        if ext == '.bag':
            if rbp:
                bag_df = rosbag_pandas.bag_to_dataframe(bag_file)
            else:
                print "No rosbag pandas cannot read bag!"

        elif ext == '.csv':
            bag_df = pd.read_csv(bag_file, parse_dates=True, index_col=0, quotechar='"')
            csv = True
        if nst is None:
            thresh = bag_df['threshold_information__data'].dropna()
        else:
            thresh = bag_df[nst + '_threshold_information__data'].dropna()

        for i in thresh.values:
            if csv:
                i = i.replace('\t', ',')
            self.handle_thresh_string(i)
        self.handle_mark_df(bag_df, ns)

    def import_thresh_file(self, thresh_file):
        """Import a previously parsed threshold file"""
        df = pd.read_csv(thresh_file, parse_dates=True, index_col=0, quotechar='"')
        columns = df.columns
        for time, values in zip(df.index, df.values):
            self.times.add(time)
            for key, val in zip(columns, values):
                if key != 'file_name' and key != 'line_number':
                    self.add_to_data(key, time, val)

    def import_mark_file(self, fname, namespace=None):
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
            bag_df = pd.read_csv(fname, parse_dates=True, index_col=0, quotechar='"')

        else:
            print "Unknown File type ", fname
        self.handle_mark_df(bag_df)

    def handle_mark_df(self, bag_df, namespace=None):
        # handle all of the marks
        if namespace is None:
            no_sec = 'mark_no_action__data_secs'
            no_nsec = 'mark_no_action__data_nsecs'
            a_sec = 'mark_action__data_secs'
            a_nsec = 'mark_action__data_nsecs'
        else:
            no_sec = namespace + '_mark_no_action__data_secs'
            no_nsec = namespace + '_mark_no_action__data_nsecs'
            a_sec = namespace + '_mark_action__data_secs'
            a_nsec = namespace + '_mark_action__data_nsecs'
        if bag_df is not None:
            if no_nsec in bag_df.columns:
                idx = bag_df[no_nsec].dropna().index
                vals = bag_df.loc[idx, [no_sec, no_nsec]]
                for _, data in vals.iterrows():
                    s = data[no_sec]
                    ns = data[no_nsec]
                    self.handle_mark(s, ns, False)
            if a_nsec in bag_df.columns:
                idx = bag_df[a_nsec].dropna().index
                vals = bag_df.loc[idx, [a_sec, a_nsec]]
                for _, data in vals.iterrows():
                    s = data[a_sec]
                    ns = data[a_nsec]
                    self.handle_mark(s, ns, True)
