#!/usr/bin/env python
# encoding: utf-8

def report(expr, line, f_name, *args,  **kwargs):
    print f_name, line, expr
    for name, val in enumerate(kwargs):
        print '\t', name, ':', val
    return True 

