#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import pprinter


class AssignFindVisitor(ast.NodeVisitor):
    '''find all of the assignments and organize them
    into global and class lists'''


    def __init__(self, sym_table):
        '''save symbol table and current class and 
        locations'''
        self.canidates = {} 
        self.canidates = {'GLOBAL_ONES' : []}
        self.current_key = 'GLOBAL_ONES'
        self.current_function = None


    def visit_ClassDef(self, node):
        '''keep track of all of the class information'''
        #set the classes
        self.current_key = node.name
        self.canidates[node.name] = []
        #visit
        self.generic_visit(node)
        #reset this stuff
        self.current_key = 'GLOBAL_ONES'


    def visit_FunctionDef(self, node):
        '''visit a function definition'''
        self.current_function = node.name
        if node.name == '__init__':
            self.generic_visit(node)
        else:
            self.generic_visit(node)
        self.current_function = None


    def visit_Assign(self, node):
        '''visit an assignment definition'''
        #we are going to look at all of the assign values here and figure out
        #if it is a constant.  Here we are just looking at __init__ for now but
        # it could be in many other location
        for i in node.targets:
            #assigning to self.asdfasfd is an attribute
            if isinstance(i, ast.Attribute):
                if isinstance(i.value, ast.Name):
                    if i.value.id == 'self':
                        #class value save it here
                        self.canidates[self.current_key].append(i.attr)
                    else:
                        print 'not assigning to self'
            elif isinstance(i, ast.Name):
                self.canidates[self.current_key].append(i.id)
            else:
                print 'ERROR not implemented type'
        self.generic_visit(node)


class FindOnlyOnceVisitor(ast.NodeVisitor):
    '''visitor that has a list of contenders and
    checks to see if they are assigned to outside of the init or 
    some other function in the code'''


    def __init__(self, canidates):
        '''save canidates  and set the current key'''
        self.canidates = canidates 
        self.current_key = 'GLOBAL_ONES'
        self.current_function = None


    def visit_ClassDef(self, node):
        #set the classes
        self.current_key = node.name
        self.generic_visit(node)
        #reset this stuff
        self.current_key = 'GLOBAL_ONES'


    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None

    
    def handle_target(self, target):
        #assigning to self.asdfasfd is an attribute
        name = None
        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name):
                if target.value.id == 'self':
                    #class value save it here
                    name = target.attr
                else:
                    print 'not assigning to self'
        elif isinstance(target, ast.Name):
            name = target.id
        else:
            print 'ERROR not implemented type'

        #as of right now if it is assigned to in another function besides 
        #init we will knock it out and continue on
        if self.current_function != '__init__':
            if name in self.canidates[self.current_key] and self.current_key != 'GLOBAL_ONES':
                print 'REMOVED: ', name
                new_list = [i for i in self.canidates[self.current_key] if i != name]
                self.canidates[self.current_key] = new_list
            elif self.current_key != 'GLOBAL_ONES':
                if name in self.canidates['GLOBAL_ONES']:
                    new_list = [i for i in self.canidates['GLOBAL_ONES'] if i != name]
                    self.canidates['GLOBAL_ONES'] = new_list


    def visit_AugAssign(self, node):
        self.handle_target(node.target)


    def visit_Assign(self, node):
        #we are going to look at all of the assign values here and figure out
        #if it is a constant.  Here we are just looking at __init__ for now but
        # it could be in many other location
        for i in node.targets:
            self.handle_target(i)

        #GO on and visit everything else
        self.generic_visit(node)


def main(fname):

    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  

            table = symtable.symtable(code, fname,'exec')
            #visit to find assignments to class vairables
            visitor = AssignFindVisitor(table)
            visitor.visit(tree)
            to_search = {}
            for k,v in visitor.canidates.iteritems():
                v = list(set(v))
                to_search[k] = v

            #visit to know out assignments not in constructor or from rospy.getparam or a constant number
            #assume that any function besides init can be called more than once
            visitor = FindOnlyOnceVisitor(to_search)
            visitor.visit(tree)
            for k,v in visitor.canidates.iteritems():
                print k
                for name in v:
                    print '\t', name 


    else:
        print 'error no file'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('This is a program to find' 
        'constant thresholds in a python program'))
    parser.add_argument('file',  help='path to file')
    args = parser.parse_args()
    main(args.file)
