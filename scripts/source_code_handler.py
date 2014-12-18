__author__ = 'ataylor'


import pkgutil
import inspect


def get_src_code(cls_obj):
    fname = inspect.getsourcefile(cls_obj)
    with open(fname) as f:
        return f.readlines()


def get_code_from_pkg_class(package, cls):
    val = __import__(package)
    attr = getattr(val, cls)
    return get_src_code(attr)



if __name__ == '__main__':
    package = 'baxter_interface'
    cls = 'Head'
    for i in get_code_from_pkg_class(package, cls):
        print i


