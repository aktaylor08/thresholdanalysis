#!/usr/bin/env python

import rospy

from rosgraph_msgs.msg import Log
from std_msgs.msg import String


class CPPConversion(object):
    """This node converts the output of the ros instrumented cpp code into the format that is
    used by the analysis"""

    def __init__(self):
        rospy.init_node("Conversion")
        self.out_publisher = rospy.Publisher('/threshold_information', String, )
        self.in_topic = rospy.Subscriber('/rosout', Log, self.out_callback,)

    def out_callback(self, msg):
        if msg.msg.startswith("threshold_information"):
            data = msg.msg[22:]
            out_msg = String('{:.7f},{:s}'.format(msg.header.stamp.to_sec(), data))
            self.out_publisher.publish(out_msg)

if __name__ == '__main__':
    CPPConversion()
    rospy.spin()