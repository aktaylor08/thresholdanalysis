#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import cfg_analysis

import pprinter

from collections import defaultdict

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






#########################################################
#
#       Start of rewriteing stuff right here
#           Hopefully a bit better way to organize and
#               keep track of stuff here
#
########################################################


class PublishCall(object):
    '''this holds info on a ropsy.publish 
    function call it will contain the class,
    the function, and a reference to the node'''


    def __init__(self, cls, func, call, expr):
        self.cls = cls
        self.func = func
        self.expr = expr
        self.call = call 


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        return '(' + str(self.expr.lineno) + ' ' + str(self.expr) + ')'


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
    '''holds information about a class variable'''


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





class IfStatementVisitor(BasicVisitor):
    '''visit if statements to ID which constants are
    used in if statements'''

    def __init__(self, canidates, cfg, func_calls):
        BasicVisitor.__init__(self)
        self.canidates = canidates
        self.cfg = cfg
        self.func_calls = func_calls


    def visit_If(self, node):
        cv = ConstantVisitor(self.canidates, self.current_class, 
                self.current_function)
        cv.visit(node.test)
        if len(cv.consts) > 0:
            cfg = self.cfg[self.current_class][self.current_function]
            depends = set()
            close_graph(node, cfg.succs, depends)
            print '\n'
            for i in depends:
                #check pub -> first degree
                #check call to pub
                #maybe assign dependent variables
                pass

            


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
                            self.current_function, node, self.current_expr))



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
        #TODO finish and verifiy
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
            func_with_call = set()
            for call in calls:
                func_with_call.add(call.func)

            ifvisit = IfStatementVisitor(canidates, flow_store, calls)
            ifvisit.visit(tree)



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
