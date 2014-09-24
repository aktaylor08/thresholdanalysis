#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse


class FucnListener(ast.NodeVisitor):

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            print ast.dump(node)
            print node.id

def main(fname):
    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  
            table = symtable.symtable(code, fname,'exec')

            print table.get_name()
            print table.get_symbols()
            # FucnListener().visit(tree)
    else:
        print 'error no file'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('This is a program to find' 
        'constant thresholds in a python program'))
    parser.add_argument('file',  help='path to file')
    args = parser.parse_args()
    main(args.file)



