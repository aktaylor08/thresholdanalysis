import os
import subprocess
import warnings
from wx.lib.mixins.listctrl import CheckListCtrlMixin, ListCtrlAutoWidthMixin
import yaml
from roslib.message import get_message_class
import wx
import sys


class CheckListCtrl(wx.ListCtrl, CheckListCtrlMixin, ListCtrlAutoWidthMixin):
    def __init__(self, parent):
        wx.ListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT | wx.SUNKEN_BORDER)
        CheckListCtrlMixin.__init__(self)
        ListCtrlAutoWidthMixin.__init__(self)


class BagFrame(wx.Frame):
    def __init__(self, title, bagfile=None):
        """Constructor"""
        wx.Frame.__init__(self, None, title=title)
        self.bagfile = bagfile
        self.bag_model = None
        self.loadbutton = None
        if self.bagfile is None:
            self.panel = wx.Panel(self, wx.ID_ANY)
            self.loadbutton = wx.Button(self.panel, label='load')
            self.loadbutton.Bind(wx.EVT_BUTTON, self.load_bag)
        else:
            self.load_bag(None)

    def load_bag(self, event):
        if self.bagfile is None:
            wildcard = "Bag File (*.bag)|*.bag"
            dialog = wx.FileDialog(None, "Choose a Bag File", os.getcwd(), "", wildcard, wx.OPEN)
            if dialog.ShowModal() == wx.ID_OK:
                self.bagfile = dialog.GetPath()
            else:
                sys.exit()
            dialog.Destroy()
        self.bag_model = BagDfModel(self.bagfile)
        if self.loadbutton is not None:
            self.loadbutton.Destroy()
        self.create_info_pane()

    def create_info_pane(self):
        panel = wx.Panel(self, -1)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        leftPanel = wx.Panel(panel, -1)
        rightPanel = wx.Panel(panel, -1)

        self.list = CheckListCtrl(rightPanel)
        self.list.InsertColumn(0, 'Topic', width=140)
        self.list.InsertColumn(1, 'Messages')
        self.list.InsertColumn(2, 'Type')
        for i in self.bag_model.topics:
            index = self.list.InsertStringItem(sys.maxint,i)
            self.list.SetStringItem(index, 1, str(self.bag_model.msg_counts[i]))
            self.list.SetStringItem(index, 2, self.bag_model.topic_map[i])

        vbox2 = wx.BoxSizer(wx.VERTICAL)

        sel = wx.Button(leftPanel, -1, 'Select All', size=(100, -1))
        des = wx.Button(leftPanel, -1, 'Deselect All', size=(100, -1))
        apply = wx.Button(leftPanel, -1, 'Import', size=(100, -1))
        self.include_header = wx.CheckBox(leftPanel, -1, 'Include Headers')

        self.Bind(wx.EVT_BUTTON, self.OnSelectAll, id=sel.GetId())
        self.Bind(wx.EVT_BUTTON, self.OnDeselectAll, id=des.GetId())
        self.Bind(wx.EVT_BUTTON, self.OnApply, id=apply.GetId())

        vbox2.Add(sel, 0, wx.TOP, 5)
        vbox2.Add(des)
        vbox2.Add(apply)
        vbox2.Add(self.include_header)

        leftPanel.SetSizer(vbox2)

        vbox.Add(self.list, 1, wx.EXPAND | wx.TOP, 3)
        vbox.Add((-1, 10))
        vbox.Add((-1, 10))

        rightPanel.SetSizer(vbox)

        hbox.Add(leftPanel, 0, wx.EXPAND | wx.RIGHT, 5)
        hbox.Add(rightPanel, 1, wx.EXPAND)
        hbox.Add((3, -1))

        panel.SetSizer(hbox)
        self.OnSelectAll(None)

    def OnSelectAll(self, event):
        num = self.list.GetItemCount()
        for i in range(num):
            self.list.CheckItem(i)

    def OnDeselectAll(self, event):
        num = self.list.GetItemCount()
        for i in range(num):
            self.list.CheckItem(i, False)

    def OnApply(self, event):
        num = self.list.GetItemCount()
        args = ['bag_reader.py', '-b', self.bag_model.bagf, '-t']
        for i in range(num):
            if self.list.IsChecked(i):
                args.append(self.list.GetItemText(i))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        lines_iterator = iter(popen.stdout.readline, b"")
        for line in lines_iterator:
            print(line)

class BagDfModel(object):
    def __init__(self, bagf):
        self.bagf = bagf
        self._bag_yaml = self.get_bag_yaml(bagf)
        self.topics = self.get_topics()
        self.msg_counts = self.get_msg_count()
        self.total_msgs = sum(self.msg_counts.itervalues())
        self.topic_to_fields, self.msg_values = self.get_msg_info()
        self.topic_map = self.build_topic_map()

    @staticmethod
    def get_bag_yaml(bag_file):
        """Get uamle dict of the bag information
        by calling the subprocess -- used to create correct sized
        arrays"""
        # Get the info on the bag
        bag_info = yaml.load(subprocess.Popen(
            ['rosbag', 'info', '--yaml', bag_file],
            stdout=subprocess.PIPE).communicate()[0])
        return bag_info

    def get_msg_info(self, parse_header=True):
        """
        Get info from all of the messages about what they contain
        and will be added to the dataframe
        """
        topic_info = self._bag_yaml['topics']
        msgs = {}
        classes = {}

        for topic in self.topics:
            msg_paths = []
            msg_types = {}
            for info in topic_info:
                if info['topic'] == topic:
                    msg_class = get_message_class(info['type'])
                    if msg_class is None:
                        warnings.warn(
                            'Could not find types for ' + topic + ' skpping ')
                    else:
                        (msg_paths, msg_types) = self.get_base_fields(msg_class(), "",
                                                                      parse_header)
                    msgs[topic] = msg_paths
                    classes[topic] = msg_types
        return msgs, classes

    def get_base_fields(self, msg, prefix='', parse_header=True):
        """function to get the full names of every message field in the message"""
        slots = msg.__slots__
        ret_val = []
        msg_types = dict()
        for i in slots:
            slot_msg = getattr(msg, i)
            if not parse_header and i == 'header':
                continue
            if hasattr(slot_msg, '__slots__'):
                (subs, type_map) = self.get_base_fields(
                    slot_msg, prefix=prefix + i + '.',
                    parse_header=parse_header,
                )

                for sub in subs:
                    ret_val.append(sub)
                for k, v in type_map.items():
                    msg_types[k] = v
            else:
                ret_val.append(prefix + i)
                msg_types[prefix + i] = slot_msg
        return ret_val, msg_types

    @staticmethod
    def get_key_name(name):
        """fix up topic to key names to make them a little prettier"""
        if name[0] == '/':
            name = name[1:]
        name = name.replace('/', '.')
        return name

    def build_topic_map(self):
        tmap = {}
        for top in self.topics:
            for yt in self._bag_yaml['topics']:
                if yt['topic'] == top:
                    tmap[top] = yt['type']
        return tmap

    def get_msg_count(self):
        """
        Find the length (# of rows) in the created dataframe
        """
        msg_counts = {}
        info = self._bag_yaml['topics']
        for topic in self.topics:
            for t in info:
                if t['topic'] == topic:
                    msg_counts[topic] = t['messages']
                    break
        return msg_counts

    def get_topics(self):
        """ Returns the names of all of the topics in the bag, and prints them
            to stdout if requested
        """
        # Pull out the topic info

        names = []
        # Store all of the topics in a dictionary
        topics = self._bag_yaml['topics']
        for topic in topics:
            names.append(topic['topic'])

        return names

    def create_data_map(self, msgs_to_read):
        """
        Create a data map for usage when parsing the bag
        """
        dmap = {}
        for topic in msgs_to_read.keys():
            base_name = self.get_key_name(topic) + '__'
            fields = {}
            for f in msgs_to_read[topic]:
                key = (base_name + f).replace('.', '_')
                fields[f] = key
            dmap[topic] = fields
        return dmap

if __name__ == '__main__':
    app = wx.App(False)
    if sys.argv > 1:
        frame = BagFrame('What up?', sys.argv[1])
    else:
        frame = BagFrame('What up?',)

    frame.Show()
    app.MainLoop()

