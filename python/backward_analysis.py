#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import cfg_analysis

import pprinter

from collections import defaultdict, deque

#########################################################
#
#       Start of rewriteing stuff right here
#           Hopefully a bit better way to organize and
#               keep track of stuff here
#
########################################################

class TreeObject(object):
    ''''hold all of the information needed 
    about a cfg node in this stuff'''

    def __init__(self, cls, func, expr, node):
        self.cls = cls
        self.func = func
        self.expr = expr
        self.node = node
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.node.lineno) + ': ' + str(self.node) 

    def __eq__(self,other):
        cls = self.cls == other.cls
        func = self.func == other.func
        node = self.node == other.node
        expr = self.expr = other.expr
        return cls and func and node and expr


    def __hash__(self):
        return hash(self.cls) + hash(self.func) + hash(self.node) + hash(self.expr)


class ClassVariable(object):
    '''holds information about a class variable'''

    def __init__(self, cls, func, name, assign):
        self.cls = cls
        self.name = name
        self.assign = assign
        self.func = func

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{:s} -> {:s}'.format(self.cls.name, self.name)  

    def __eq__(self, other):
        if not isinstance(other, ClassVariable):
            return False
        cls = self.cls == other.cls
        name = self.name == other.name
        return cls and name 


    def __hash__(self):
        return hash(self.cls) and hash(self.name) 



class FunctionVariable(object):
    '''holds information about a Function variable'''


    def __init__(self, cls, func, name, assign):
        self.cls = cls
        self.func = func
        self.name = name
        self.assign = assign

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.cls, ast.Module):
            if self.func is None:
                return '{:s}.{:s} -> {:s}'.format('GLOBAL', 
                        'None', self.name)
            else:
                return '{:s}.{:s} -> {:s}'.format('GLOBAL', 
                        self.func.name, self.name)

        else:
            if self.func is None:
                return '{:s}.{:s} -> {:s}'.format(self.cls.name, 
                    'CLASS', self.name)
            else:
                return '{:s}.{:s} -> {:s}'.format(self.cls.name, 
                    self.func.name, self.name)

    def __eq__(self, other):
        func = self.func == other.func
        name = self.name == other.name
        return func and name 


    def __hash__(self):
        return hash(self.cls) and hash(self.name) 


class SearchStruct(object):
    '''data structure that holds information for the search/backward
    analysis'''


    def __init__(self, statement, publisher, child, distance):
        self.statement = statement
        self.publisher = publisher 
        self.child = child
        self.distance = distance
        self.parent = None


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.distance) + ' ' +  str(self.statement)

    def __eq__(self, other):
        return self.statement == other.statement

    def __hash__(self):
        return hash(self.statement)


class CanidateStore(object):
    '''class to hold all of the candiates for 
        thresholds.  Will include num literals, class variables,
        and variables wihtin a function'''


    def __init__(self, assignments, tree):
        '''build the list from found assignments'''
        self.assignments = assignments
        self.tree = tree
        self.class_vars = {}
        self.func_vars = {} 
        self.compile_canidates()


    def compile_canidates(self):
        '''compile all of the assignments down into a list that we can check
        to see what they really assign and how many times they are assigned'''
        self.do_class_variables()
        self.do_func_variables()

    def do_class_variables(self):
        '''Define all of the class variables as constants or not'''
        #book keeping
        for_certain = set()
        bad = set()
        init = set()
        elsewhere = set()
        maybe = set()
        classes = self.assignments.keys()

        for cls in classes:
            variables = sorted(self.assignments[cls], key=lambda x: x.func.name)
            for i in variables:
                if isinstance(i, ClassVariable):
                    if isinstance(i.assign, ast.AugAssign):
                        bad.add(i)
                    else:
                        #if it makes a call to rospy.get_param it is a threshold
                        if isinstance(i.assign.value, ast.Call):
                            if self.is_paramcall(i.assign.value):
                                for_certain.add(i)

                        #as of right now we just increment in init but
                        if i.func.name == '__init__':
                            init.add(i)
                        else:
                            if self.check_only_const(i):
                                maybe.add(i)
                            else:
                                elsewhere.add(i)

        vals = init.union(maybe).difference(elsewhere).difference(bad).union(for_certain)
        for i in vals:
            if i.cls in self.class_vars:
                self.class_vars[i.cls].append(i)
            else:
                self.class_vars[i.cls] = [i]



    def do_func_variables(self):
        '''Define function variables as constants or not'''
        classes = self.assignments.keys()
        canidates = {}
        bad = bad = set()
        for cls in classes:
            variables = sorted(self.assignments[cls], key=lambda x: x.name)
            for i in variables:
                if isinstance(i, FunctionVariable):
                    if isinstance(i.assign, ast.AugAssign):
                        bad.add(i)
                    else:
                        const = self.check_only_const(i.assign.value)
                        if const:
                            if i in canidates:
                                canidates[i] +=1
                            else:
                                canidates[i] = 1
                        else:
                            bad.add(i)
        vals = [x for x in canidates.keys() if canidates[x] == 1]
        vals = set(vals)
        vals = vals - bad
        for i in vals:
            if i.cls in self.func_vars :
                self.func_vars[i.cls].append(i)
            else:
                self.func_vars[i.cls] = [i]


    def check_only_const(self, node):
        if isinstance(node, ast.Num):
            return True
        elif isinstance(node, ast.BinOp):
            left = self.check_only_const(node.left) 
            right = self.check_only_const(node.right)
            return left and right
        elif isinstance(node, ast.Call):
            #check to see if it is a call to what is a constant param
            return self.is_paramcall(node)
        elif isinstance(node, ast.Attribute):
            return self.is_const(node)
    
    
    def is_paramcall(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'rospy' and node.func.attr == 'get_param':
                    return True
        return False


    def is_const(self, node):
        '''given a node of an ast return if it is
        a constant candidate -> looks up name and  other stuff'''
        cls, func = self.get_class_and_func(node)

        #if it is an attribute than check if it is a self call and
        #htan check to see if it is in its class listing
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self': 
                if cls in self.class_vars:
                    if node.attr in self.class_vars[cls]:
                        return True
            return False


    def get_class_and_func(self, node):
        visitor = ClassFuncVisit(node)
        visitor.visit(self.tree)
        return visitor.cls, visitor.func



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
        self.current_expr = None


    def visit_Module(self, node):
        '''we set the module level as the current class'''
        self.current_class = node
        self.current_function = ast.FunctionDef('GLOBAL_FUNCTION', [], [], []) 
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


    def visit_Expr(self, node):
        self.current_expr = node
        self.generic_visit(node)
        self.current_expr = None


class ConstantVisitor(BasicVisitor):

    def __init__(self, canidates, cls, func):
        BasicVisitor.__init__(self)
        self.canidates = canidates
        self.consts =  []
        self.current_class = cls
        self.current_function = func

    def visit_Num(self, node):
        self.consts.append(node)


    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id == 'self':
                cv = ClassVariable(self.current_class, self.current_function, 
                    node.attr, node)
                if cv in self.canidates.class_vars[self.current_class]:
                    self.consts.append(cv)

    def visit_Name(self, node):
            fv = FunctionVariable( self.current_class, 
                    self.current_function, node.id, node)
            if self.current_class in self.canidates.func_vars:
                if fv in self.canidates.func_vars[self.current_class]:
                    self.consts.append(fv)


class AssignFindVisitor(BasicVisitor):
    '''find all of the assignments and organize them
    into global and class lists'''


    def __init__(self):
        '''save symbol table and current class and 
        locations'''
        BasicVisitor.__init__(self)
        self.canidates =  defaultdict(list)


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
                        self.canidates[self.current_class].append(ClassVariable(
                            self.current_class, self.current_function, i.attr, node ))
                    else:
                        print 'not assigning to self'

            elif isinstance(i, ast.Name):
                self.canidates[self.current_class].append(FunctionVariable(
                    self.current_class, self.current_function, i.id, node))

            else:
                print 'ERROR not implemented type'

        self.generic_visit(node)


    def visit_AugAssign(self, node):

        i =  node.target
        #assigning to self.asdfasfd is an attribute
        if isinstance(i, ast.Attribute):
            if isinstance(i.value, ast.Name):
                if i.value.id == 'self':
                    #class value save it here
                    self.canidates[self.current_class].append(ClassVariable(
                        self.current_class, self.current_function, i.attr, node ))
                else:
                    print 'not assigning to self'

        elif isinstance(i, ast.Name):
            self.canidates[self.current_class].append(FunctionVariable(
                self.current_class, self.current_function, i.id, node))

        else:
            print 'ERROR not implemented type'
        self.generic_visit(node)



class IfOrFuncVisitor(BasicVisitor):
    '''finds if the program is part of an if statement'''

    def __init__(self, target):
        BasicVisitor.__init__(self)
        self.target = target
        self.canidates = deque()
        self.res = None

    def visit_FunctionDef(self, node):
        self.canidates.appendleft(node)
        BasicVisitor.visit_FunctionDef(self, node)
        self.canidates.popleft()


    def visit_If(self, node):
        self.canidates.appendleft(node)
        self.generic_visit(node)
        self.canidates.popleft()

    def generic_visit(self, node):
        popped = []
        if node == self.target:
            found = False
            while not found:
                temp = self.canidates.popleft()
                popped.append(temp)
                if temp == node:
                    continue
                else:
                    found = True
                    self.res = TreeObject(self.current_class, self.current_function, 
                            self.current_expr, temp)
                    break
            for i in reversed(popped):
                self.canidates.append(popped)
        else:
            BasicVisitor.generic_visit(self, node)


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
                        TreeObject(self.current_class, 
                            self.current_function, self.current_expr, node))


class ClassFuncVisit(BasicVisitor):

    def __init__(self, target):
        BasicVisitor.__init__(self)
        self.target = target
        self.cls = None
        self.func = None


    def generic_visit(self, node):
        if node == self.target:
            self.cls = self.current_class
            self.func = self.current_function
        else:
            BasicVisitor.generic_visit(self, node)


class BackwardAnalysis(object):
    '''class to perform the backward analysis needed on all of the files'''

    def __init__(self, canidates, calls, flow_store, tree):
        self.canidates = canidates
        self.calls = calls
        self.flow_store = flow_store
        self.tree = tree
        self.if_visitor = IfConstantVisitor(self.canidates)
        self.if_visitor.visit(tree)


    def compute(self):
        searched = set()
        to_search = deque()
        thresh = {} 
        for call in self.calls:
            obj = SearchStruct(call,call, None, 0)
            to_search.append(obj)

        while len(to_search) > 0:
            current = to_search.popleft()
            #find some thresholds
            new_thresholds = self.find_thresholds(current)
            if len(new_thresholds) > 0:
                thresh[current] = new_thresholds
            #get data flows from here
            new_data = self.find_data_dependiences(current)
            #get new flow dependinces here
            new_flow = self.find_flow_dependencies(current)

            for can in new_data:
                if can in searched:
                    #do some math here to make sure its not less
                    print 'already searched?'
                else:
                    print 'adding canidate'
                    to_search.append(can)

            for can in new_flow:
                if can in searched:
                    #do some math here to make sure its not less
                    print 'already searched?:', can
                else:
                    to_search.append(can)
            searched.add(current)
        to_print = sorted(list(searched), key=lambda x: x.distance)
        for i in to_print:
            if i in thresh:
                print '\n'
                print 'Thresholds: ', thresh[i]
                full_print(i)
    
    def find_thresholds(self, current):
        '''find any thresholds in the current statement and 
        return them if we find any'''
        if current.statement.node in self.if_visitor.ifs:
            return self.if_visitor.ifs[current.statement.node] 
        else:
            return [] 


    def find_data_dependiences(self, current):
        '''find any thresholds in the current statement and 
        return them if we find any'''
        #TODO Implement
        return []


    def find_flow_dependencies(self, current):
        '''find flow dependencies'''
        visitor = IfOrFuncVisitor(current.statement.node)
        visitor.visit(self.tree)
        to_return = []
        if isinstance(visitor.res.node, ast.If):
            obj = SearchStruct(visitor.res, current.publisher, current, current.distance + 1)
            to_return.append(obj)
        else:
            #otherwise search for function calls here?
            for call in self.search_function_calls(visitor.res):
                obj = SearchStruct(call, current.publisher, current, current.distance + 1)
                to_return.append(obj)

        return to_return 

    def search_function_calls(self, tree_thing):
        fcv = FindCallVisitor(tree_thing.cls, tree_thing.func)
        fcv.visit(self.tree)
        return fcv.calls


def full_print(obj, tabs=0):
    print '\t' * tabs, obj
    if obj.child is not None:
        full_print(obj.child, tabs+1)


class  FindCallVisitor(BasicVisitor):

    def __init__(self, target_class, target_func):
        BasicVisitor.__init__(self)
        self.target_class = target_class
        self.target_func = target_func
        self.calls = []


    def visit_Call(self, node):
        if self.current_class == self.target_class:
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == 'self' and node.func.attr == self.target_func.name:
                        #save this for use
                        self.calls.append(TreeObject(self.current_class, 
                                self.current_function, node, self.current_expr))


class IfConstantVisitor(BasicVisitor):
    '''visit if statements to ID which constants are
    used in if statements'''

    def __init__(self, canidates):
        BasicVisitor.__init__(self)
        self.canidates = canidates
        self.ifs = {}


    def visit_If(self, node):
        cv = ConstantVisitor(self.canidates, self.current_class, 
                self.current_function)
        cv.visit(node.test)
        if len(cv.consts) > 0:
            self.ifs[node] =  cv.consts
        self.generic_visit(node)

    def __repr__(self):
        return self.__str__()


    def __str__(self):
        string = ''
        for i in self.ifs:
            string += str(i.lineno) + ' ' + str(i) + ':\n'
            for const in self.ifs[i]:
                string += '\t' + str(const) + '\n'

        return string


        
        


class ConstantVisitor(BasicVisitor):
    '''IDs constants from candidates and also numberical constants'''

    def __init__(self, canidates, cls, func):
        BasicVisitor.__init__(self)
        self.canidates = canidates
        self.consts =  []
        self.current_class = cls
        self.current_function = func

    def visit_Num(self, node):
        self.consts.append(node)


    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id == 'self':
                cv = ClassVariable(self.current_class, self.current_function, 
                    node.attr, node)
                if cv in self.canidates.class_vars[self.current_class]:
                    self.consts.append(cv)

    def visit_Name(self, node):
            fv = FunctionVariable( self.current_class, 
                    self.current_function, node.id, node)
            if self.current_class in self.canidates.func_vars:
                if fv in self.canidates.func_vars[self.current_class]:
                    self.consts.append(fv)

def analyze_file(fname):
    '''new main function...get CFG and find pubs first'''
    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  

            a = AssignFindVisitor()
            a.visit(tree)
            canidates = CanidateStore(a.canidates, tree)

            flow_store = cfg_analysis.build_files_cfgs(tree=tree)
            publish_finder = PublishFinderVisitor()
            publish_finder.visit(tree)
            calls = publish_finder.publish_calls
            ba = BackwardAnalysis(canidates, calls, flow_store, tree)
            ba.compute()


    else:
        print 'error no file'


def close_graph(node, graph, visited):
    if node in visited:
        return
    if node not in graph:
        return
    visited.add(node)
    for target in graph[node]:
        close_graph(target, graph, visited)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('This is a program to find' 
        'constant thresholds in a python program'))
    parser.add_argument('file',  help='path to file')
    args = parser.parse_args()
    analyze_file(args.file)
