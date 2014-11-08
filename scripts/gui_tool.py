#!/usr/bin/python

# Simply gui program to monitor the topics of interest
# during flight using the gui fly node of mitAscTect
import rospy
from std_msgs.msg import Time
 
import gtk
import gobject
import threading
import signal

#global values shared between the two classes here these will be updated by the callbacks 
lock = threading.Lock()

gobject.threads_init()

class CtrlNode:

    def __init__(self):
        #do all ros stuff
        rospy.init_node('gui_tool')
        self.bad_pub = rospy.Publisher('mark_bad', Time)
        self.good_pub = rospy.Publisher('mark_good', Time)

    

class PyApp(gtk.Window):
    '''class defining all of the widgets and stuff'''

    def __init__(self):

        super(PyApp, self).__init__()



        self.set_position(gtk.WIN_POS_CENTER)
        self.set_border_width(8)
        self.connect("destroy", gtk.main_quit)
        self.set_title("Marker")
        
        #Vertical Orginzer 
        vert_origizer = gtk.VBox(False, 10)

    
        self.bad_button = gtk.Button("Bad")
        self.bad_button.connect("clicked", self.on_bad_clicked)
        box = gtk.HBox(False, 5)
        box.add(self.bad_button)
        vert_origizer.add(box)

        self.good_button = gtk.Button("Good")
        self.good_button.connect("clicked", self.on_good_clicked)
        box = gtk.HBox(False, 5)
        box.add(self.good_button)
        vert_origizer.add(box)

        #final additons
        self.add(vert_origizer)
        self.show_all() 
        self.node = CtrlNode()
        # gobject.timeout_add(500, self.update_info)


    def on_bad_clicked(self, widget):
        self.node.bad_pub.publish(rospy.Time.now())

    def on_good_clicked(self, widget):
        self.node.good_pub.publish(rospy.Time.now())


if __name__ == '__main__':
    app =  PyApp()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    gtk.main()