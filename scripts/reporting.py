#!/usr/bin/env python
# encoding: utf-8

import rospy
import binascii
import datetime
from std_msgs.msg import String


def report(expr, f_name, line, check, *args,  **kwargs):

    vals = ['{:.7f}'.format(rospy.Time.now().to_sec()), str(
        f_name), str(line), str(check), str(expr)]
    for key in kwargs:
        if type(kwargs[key]) is bytes:
            vals.append(str(key) + ':' + binascii.hexlify(kwargs[key]))
        else:
            vals.append(str(key) + ':' + str(kwargs[key]))
    vals = ','.join(vals)
    Reporter.Instance().publish(vals)
    return expr


class Singleton:

    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class Reporter:

    def __init__(self):
        print rospy.get_name()
        self.pub = rospy.Publisher(
            'threshold_information', String,) #queue_size=100)

    def publish(self, msg):
        self.pub.publish(msg)
