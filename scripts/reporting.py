#!/usr/bin/env python
# encoding: utf-8

import rospy
import binascii
import datetime
from std_msgs.msg import String


def report(filename, lineno, result, arg_dict, *args, **kwargs):
    vals = ['{:.7f}'.format(rospy.Time.now().to_sec()), str(
        filename), str(lineno)]
    for key in kwargs:
        if key.startswith('res_'):
            d = {i : kwargs[i] for i in arg_dict[key]}
            kwargs[key] = kwargs[key](**d)
        if type(kwargs[key]) is bytes:
            kwargs[key] + ':' + binascii.hexlify(kwargs[key])
    d = {i : kwargs[i] for i in arg_dict['result']}
    result = result(**d)
    vals.append(str(result))
    for key in kwargs:
        vals.append(str(kwargs[key]))
    vals = ','.join(vals)
    Reporter.Instance().publish(vals)
    return result


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
        self.pub = rospy.Publisher(
            '/threshold_information', String,) #queue_size=100)

    def publish(self, msg):
        self.pub.publish(msg)
