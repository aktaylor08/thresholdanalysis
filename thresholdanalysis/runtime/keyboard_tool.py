#!/usr/bin/python

# Simply gui program to monitor the topics of interest
# during flight using the gui fly runtime of mitAscTect
import rospy
from std_msgs.msg import Time
from keyboard.msg import Key




class KeyboardMarker(object):

    def __init__(self):
        # do all runtime stuff
        rospy.init_node('gui_tool')
        rospy.Subscriber('keyboard/keydown', Key, self.key_callback)
        mark_advance = rospy.get_param('advance_keys', [97, 103, 108])
        mark_no_advance = rospy.get_param('no_advance_keys', [122, 98, 46])
        self.no_adv = {}
        self.adv = {}
        no = '/marks{:d}/mark_no_action'
        yes = '/marks{:d}/mark_action'
        for idx, key in enumerate(mark_advance):
            self.adv[key] = rospy.Publisher(yes.format(idx), Time)
        for idx, key in enumerate(mark_no_advance):
            self.no_adv[key] = rospy.Publisher(no.format(idx), Time)

    def key_callback(self, msg):
        if msg.code in self.adv:
            self.adv[msg.code].publish(rospy.Time.now())
            rospy.logerr('Advance when should not marked by %d', msg.code);
        if msg.code in self.no_adv:
            self.no_adv[msg.code].publish(rospy.Time.now())
            rospy.logerr('No Advance marked by %d', msg.code);


if __name__ == '__main__':
    KeyboardMarker()
    rospy.spin()


