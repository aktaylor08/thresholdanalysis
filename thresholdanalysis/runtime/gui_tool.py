#!/usr/bin/python

# Simply gui program to monitor the topics of interest
# during flight using the gui fly runtime of mitAscTect
import rospy
from std_msgs.msg import Time

import gtk
import gobject
import threading
import signal

# global values shared between the two classes here these will be updated by the callbacks
lock = threading.Lock()

gobject.threads_init()


class CtrlNode:
    def __init__(self):
        # do all runtime stuff
        rospy.init_node('gui_tool')
        self.bad_pub = rospy.Publisher('mark_no_action', Time, queue_size=10)
        self.good_pub = rospy.Publisher('mark_action', Time, queue_size=10)


class PyApp(gtk.Window):
    """class defining all of the widgets and stuff"""

    def __init__(self):
        super(PyApp, self).__init__()

        self.set_position(gtk.WIN_POS_CENTER)
        self.set_border_width(8)
        self.connect("destroy", gtk.main_quit)
        self.set_title("Marker")

        # Vertical Organizer
        vert_origizer = gtk.VBox(False, 10)

        self.bad_button = gtk.Button("Should Be Doing Something")
        self.bad_button.connect("clicked", self.on_bad_clicked)
        box = gtk.HBox(False, 5)
        box.add(self.bad_button)
        vert_origizer.add(box)

        self.good_button = gtk.Button("Did something when shouldn't")
        self.good_button.connect("clicked", self.on_good_clicked)
        box = gtk.HBox(False, 5)
        box.add(self.good_button)
        vert_origizer.add(box)

        # final additions
        self.add(vert_origizer)
        self.show_all()
        self.node = CtrlNode()

    def on_bad_clicked(self, _):
        self.node.bad_pub.publish(rospy.Time.now())

    def on_good_clicked(self, _):
        self.node.good_pub.publish(rospy.Time.now())

if __name__ == '__main__':
    app = PyApp()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    gtk.main()
