#!/usr/bin/env python
# encoding: utf-8

def report(expr, line, f_name, **kwargs):
    print f_name, line, expr
    for name, val in enumerate(kwargs):
        print name, ':', val
    return True 

