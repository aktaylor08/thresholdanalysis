#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import cfg

import pprinter

class FunctionInfo(object):
    '''representation of a function here'''

    def __init__(self, cls=str(), func=str()):
        '''create a function info option'''
        self.cls = cls 
        self.func = func 

    def __repr__(self):
        '''make a string'''
        return '(class: {:s} function: {:s})'.format(
                self.cls, self.func)

    def __str__(self):
        '''make a string'''
        return self.__repr__()

    def __eq__(self, other):
        '''check equality'''
        val = self.cls == other.cls and self.func == other.func
        return val 

    def __hash__(self):
        return hash(self.cls + self.func)
        

class FunctionCallGraph(object):
    '''holds a graph of the function calls'''
    

    def __init__(self):
        self._graph = {}

    def add_call(self, func, target):
        '''
        Add a call to the list
        '''
        if func in self._graph:
            self._graph[func].add(target)
        else:
            self._graph[func] = set()
            self._graph[func].add(target)

    def get_calls(self, func):
        '''Return the calls from a function or
        None if it doesn't exisit'''
        if func in self._graph:
            return list(self._graph[func])
        else:
            return None


    def get_dict_rep(self):
        '''get a dictonary representation of this map'''
        d = dict()
        for k,v in self._graph.iteritems():
            d[k] = list(v)
        return d


class ConstVariable(object):
    '''constant variable not used at the moment'''

    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __repr__(self):
        return '(class: {:s} name: {:s})'.format(
                self.cls, self.name)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return other.cls == self.cls and other.name == self.name

    def __hash__(self):
        return hash(self.cls + self.name)


class AssignFindVisitor(ast.NodeVisitor):
    '''find all of the assignments and organize them
    into global and class lists'''


    def __init__(self):
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
                # print 'REMOVED: ', name
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

        #get param might appear somewhere else so don't get rid of this
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and isinstance(func.attr, str):
                    if func.value.id == 'rospy' and func.attr == 'get_param':
                        self.generic_visit(node)
                        return 

        #TODO: Don't get rid of constant only expressions on the assign side?

        #otherwise get rid fo some stuff
        for i in node.targets:
            self.handle_target(i)
            

        #GO on and visit everything else
        self.generic_visit(node)


class FunctionGraphVisitor(ast.NodeVisitor):
    ''' Create a function'''

    def __init__(self):
        self.func_map = FunctionCallGraph() 
        self.current_class = 'GLOBAL_OBJECTS'
        self.current_function= None 
        

    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None 


    def visit_ClassDef(self, node):
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = 'GLOBAL_OBJECTS' 

    def visit_Module(self, node):
        print node
        self.generic_visit(node)

    
    def visit_Call(self, node):
        #See if we are calling another function in here
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'self':
                    key = FunctionInfo(cls=self.current_class,
                            func = self.current_function)
                    value = FunctionInfo(cls=self.current_class,
                            func=node.func.attr)
                    self.func_map.add_call(key,value)




class IfStatementVisitor(ast.NodeVisitor):
    '''visit if statements to ID which constants are
    used in if statements'''

    def __init__(self, values):
        self.to_search = values
        self.thresh = set()
        self.in_test = False
        self.current_function = None
        self.current_class = 'GLOBAL_ONES'


    def visit_ClassDef(self, node):
        #set the classes
        self.current_class = node.name
        self.generic_visit(node)
        #reset this stuff
        self.current_class = 'GLOBAL_ONES'


    def visit_FunctionDef(self, node):
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None

    def visit_Attribute(self, node):
        if self.in_test:
            #check to see if we have a comparision agains a value here
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                val = ConstVariable(cls=self.current_class, 
                        name=node.attr)
                if val in self.to_search:
                    self.thresh.add(val)
                
    def visit_Name(self, node):
        if self.in_test:
            val =  ConstVariable(cls='GLOBAL_ONES', name=node.id)
            if val in self.to_search:
                self.thresh.add(val)



    def visit_If(self, node):
        self.in_test = True
        self.visit(node.test)
        self.in_test = False
        for i in node.body:
            self.visit(i)
        for i in node.orelse:
            self.visit(i)


def main(fname):
    '''main function'''

    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  

            #visit to find assignments to class vairables
            visitor = AssignFindVisitor()
            visitor.visit(tree)
            to_search = {}
            for k,v in visitor.canidates.iteritems():
                v = list(set(v))
                to_search[k] = v

            #visit to know out assignments not in constructor or from rospy.getparam or a constant number
            #assume that any function besides init can be called more than once
            visitor = FindOnlyOnceVisitor(to_search)
            visitor.visit(tree)
            canidates = []
            for k,v in visitor.canidates.iteritems():
                for name in v:
                    canidates.append(ConstVariable(cls=k, name=name))
            print 'Canidates:'
            for i in  canidates:
                print '\t', i

            visitor = PublishFinderVisitor()
            visitor.visit(tree)
            pub_calls = visitor.publish_calls

            visitor = FunctionGraphVisitor()
            visitor.visit(tree)
            func_map = visitor.func_map.get_dict_rep()

            visitor = IfStatementVisitor(canidates)
            visitor.visit(tree)
            thresholds = list(visitor.thresh)
            print 'Thresholds:'
            for i in thresholds:
                print '\t', i

    else:
        print 'error no file'





#########################################################
#
#       Start of rewriteing stuff right here
#           Hopefully a bit better way to organize and
#               keep track of stuff here
#
######################################################


class PublishCall(object):
    '''this holds info on a ropsy.publish 
    function call it will contain the class,
    the function, and a reference to the node'''

    def __init__(self, cls, func, node):
        self.cls = cls
        self.func = func
        self.node = node

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '(' + str(self.node.lineno) + ' ' + str(self.node) + ')'

    def __eq__(self,other):
        cls = self.cls == other.cls
        func = self.func == other.func
        node = self.node == self.node
        return cls and func and node

    def __hash__(self):
        return hash(self.cls) + hash(self.func) + hash(self.node)



class BasicVisitor(ast.NodeVisitor):
    '''this is a super simple visitor 
    which keeps track of the current class
    and function that you are in while traversing 
    the tree.  Can be extended to keep the functionality
    without having to copy a bunch of code'''

    def __init__(self):
        '''start the tracking'''
        self.current_class = None 
        self.current_function = None


    def visit_Module(self, node):
        '''we set the module level as the current class'''
        self.current_class = node
        self.generic_visit(node)
        self.current_class = None


    def visit_FunctionDef(self, node):
        '''do some assignments'''
        old_func = self.current_function 
        self.current_function = node
        self.generic_visit(node)
        self.current_function = old_func 


    def visit_ClassDef(self, node):
        '''do some more assingments'''
        old_class =  self.current_class
        self.current_class = node
        self.generic_visit(node)
        self.current_class = old_class 


class PublishFinderVisitor(BasicVisitor):
    '''find and store all of the rospy.publish calls 
    in this manner we can get all of the functions and 
    stuff that they reside in  will store them in an object'''

    def __init__(self):
        BasicVisitor.__init__(self)
        self.publish_calls = []

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            #skipping for now
            pass
        elif isinstance(func, ast.Attribute):
            if func.attr == 'publish':
                self.publish_calls.append(
                        PublishCall(self.current_class, 
                            self.current_function, node))


def analyze_file(fname):
    '''new main function...get CFG and find pubs first'''
    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  

            flow_store = cfg.build_files_cfgs(tree=tree)
            publish_finder = PublishFinderVisitor()
            publish_finder.visit(tree)
            calls = publish_finder.publish_calls
            for call in calls:
                cfg_g = flow_store[call.cls][call.func]
                cfg.print_graph(cfg_g.preds)
                print cfg_g.init_map
                print call.node

                

    else:
        print 'error no file'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('This is a program to find' 
        'constant thresholds in a python program'))
    parser.add_argument('file',  help='path to file')
    args = parser.parse_args()
    analyze_file(args.file)
