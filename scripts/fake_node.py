#!/usr/bin/env python

import rospy
import random

from std_msgs.msg import Int16


class Faker(object):

    def __init__(self):
        rospy.init_node("FAKEKTR")
        self.publiser = rospy.Publisher("/give_me_info",  Int16)

    def main_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            val = random.randint(0,60)
            self.publiser.publish(val)
            rate.sleep()

if __name__ == "__main__":
    Faker().main_loop()
