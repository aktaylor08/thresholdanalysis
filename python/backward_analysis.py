#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import cfg_analysis

import pprinter

from collections import defaultdict, deque


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


    def __init__(self, statement, publisher, children, distance, important=False, distance_cost=1):
        self.statement = statement
        self.publisher = publisher 
        if children is None:
            self.children = []
        elif not isinstance(children, list):
            self.children = [children]
        else:
            self.children= children
        self.distance = distance
        self.parent = None
        self.important = important


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.distance) + ' ' +  str(self.statement)

    def __eq__(self, other):
        return self.statement.node == other.statement.node

    def __hash__(self):
        return hash(self.statement.node)


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



class ReachingDefinition(object):

    '''class to compute reaching definitions on all functions within
    a file.  Will compute both exit and enter values for all of them'''



    def __init__(self, tree, cfg_store):
        '''build'''
        self.tree = tree
        self.cfg_store = cfg_store
        self.rds_in = {}
        self.rds_out = {}

    def compute(self):
        '''compute RD for each funciton'''
        for i in self.cfg_store:
            self.rds_out[i] = {}
            self.rds_in[i] = {}
            for func in self.cfg_store[i]:
                ins, outs= self.do_function(self.cfg_store[i][func])
                self.rds_out[i][func] = outs 
                self.rds_in[i][func] = ins 




    def do_function(self, cfg):
        '''compute ins and outs for a function
        start off with any params that are not self in the function
        and than do some iteration until you reach a fix point'''
        outs = {}
        ins = {}
        for i in cfg.preds:
            outs[i] = set() 
        func = list(cfg.preds[cfg.start])[0]
        #handle the arguments
        arguments = func.args.args
        for arg in arguments:
            if isinstance(arg, ast.Name):
                if arg.id == 'self':
                    pass
                else:
                    outs[func].add((arg.id, arg))
            else:
                print 'ERROROROR line 325ish'

        #now we will iterate until something changes
        changed = True 
        while changed:
            seen = set()
            node = cfg.start
            changed = self.iterate(seen, node, outs, cfg)

        #ins are just the union of the preceeding outs.  
        for i in outs:
            if isinstance(i, ast.FunctionDef):
                continue
            preds = cfg.preds[i]
            vals = set()
            for p in preds:
               for o in outs[p]:
                   vals.add(o)
            ins[i] = vals
        return ins, outs




    def iterate(self, seen, node, outs, cfg):
        '''this is the main function that computs gens
        and kills and than does the union on the entering data'''
        if node in seen:
            return False
        if isinstance(node, ast.FunctionDef):
            return False

        changed = False
        #add intials
        vals = cfg.preds[node]
        ins = set()
        for  val in vals:
            for to_add in outs[val]:
                ins.add(to_add)

        #gen kill set operations
        gen = self.get_gen(node)
        kill = self.get_kill(node, ins)
        temp = (ins - kill)
        for one_gen in gen:
            temp.add(one_gen)
        if temp !=  outs[node]:
            changed = True
            outs[node] = temp

        #keep track
        seen.add(node)
        #visit all the successors
        if node in cfg.succs:
            for i in cfg.succs[node]:
                changed = self.iterate(seen, i, outs, cfg) or changed
        return changed


    def get_kill(self, node, current):
        '''kill set -> here its any assignment'''
        to_return = set()
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = get_name(target, '')

        elif isinstance(node, ast.AugAssign):
            target = node.target
            name = get_name(target, '')
        else:
            return set()

        for val in current:
            if val[0] == name:
                to_return.add(val)
        return to_return


    def get_gen(self, node):
        '''gen set -> any assignment'''
        to_return = set()
        if isinstance(node, ast.Assign):
            for target in node.targets:
                to_return.add((get_name(target),node))
        elif isinstance(node, ast.AugAssign):
            target = node.target
            to_return.add((get_name(target),node))
        return to_return



def get_name(attr, start=str()):
    '''get the name recursivley defined'''
    if isinstance(attr, ast.Name):
        name = attr.id
    elif isinstance(attr, ast.Attribute):
        name = get_name(attr.value, start) +'.' +  get_name(attr.attr, start)
    elif isinstance(attr, str):
        name =  attr 
    else:
        name = ''
    return name


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

    def generic_visit(self, node):
        if isinstance(node, ast.If):
            self.current_expr = node
            ast.NodeVisitor.generic_visit(self, node)
            self.current_expr = None
        else:
            ast.NodeVisitor.generic_visit(self, node)




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
                        # print ast.dump(node)
                        #TODO Do we need to worry about anyting else?
                        pass

            elif isinstance(i, ast.Name):
                self.canidates[self.current_class].append(FunctionVariable(
                    self.current_class, self.current_function, i.id, node))

            else:
                print 'ERROR not implemented type:', node.lineno, type(node)

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

class GetVarsVisit(ast.NodeVisitor):

    def __init__(self, statement):
        self.statement = statement
        self.class_vars = set()
        self.func_vars = set()

    def visit_Name(self, node):
        '''add it to the function variables'''
        self.func_vars.add(node.id)


    def visit_Attribute(self, node):
        '''if it is an attribute check the name
        to see if it starts with self'''
        name = get_name(node)
        self.func_vars.add(name)
        if name.startswith('self.'):
           #we have a class variable so check it out
           cv = ClassVariable(self.statement.cls, self.statement.func,
                name, node)
           self.class_vars.add(cv)
            



class FindAssigns(BasicVisitor):

    def __init__(self, var):
        BasicVisitor.__init__(self)
        self.var = var
        self.assignments = [] 


    def visit_Assign(self, node):
        '''visit an assignment definition'''

        #we are going to look at all of the assign values here and figure out
        #if it is a constant.  Here we are just looking at __init__ for now but
        # it could be in many other location
        for i in node.targets:
            #assigning to self.asdfasfd is an attribute
            name = get_name(i)
            names = name.split('.')
            varsplit = self.var.name.split('.')
            if name.startswith('self.') and names ==varsplit[:len(names)]:
                if self.current_function != self.var.func:
                    self.assignments.append(TreeObject(self.current_class, 
                       self.current_function, self.current_expr, node))


    def visit_AugAssign(self, node):

        i =  node.target
        #assigning to self.asdfasfd is an attribute
        name = get_name(i)
        names = name.split('.')
        varsplit = self.var.name.split('.')
        if name.startswith('self.') and names ==varsplit[:len(names)]:
            if self.current_function != self.var.func:
                self.assignments.append(TreeObject(self.current_class, 
                        self.current_function, self.current_expr, node))





class BackwardAnalysis(object):
    '''class to perform the backward analysis needed on all of the files'''

    def __init__(self, canidates, calls, flow_store, tree, reaching_defs, verbose=False, web=False):
        self.canidates = canidates
        self.calls = calls
        self.flow_store = flow_store
        self.tree = tree
        self.if_visitor = IfConstantVisitor(self.canidates)
        self.if_visitor.visit(tree)
        self.reaching_defs = reaching_defs
        self.verbose = verbose
        self.web_style = web
        self.thresholds = []


    def compute(self):
        searched = set()
        to_search = deque()
        thresh = {} 
        #add important statements
        for call in self.calls:
            obj = SearchStruct(call,call, None, 0, important=True, distance_cost=0)
            to_search.append(obj)
        while len(to_search) > 0:
            current = to_search.popleft()
            if self.verbose:
                print '\n'
                print current
            #find some thresholds
            new_thresholds = self.find_thresholds(current)
            if len(new_thresholds) > 0:
                thresh[current] = new_thresholds
                if self.verbose:
                    print '\tFOUND THRESHOLD!:',
                for i in new_thresholds:
                    pass
                    if self.verbose:
                        print   i,
                        print 
            #get data flows from here
            new_data = self.find_data_dependiences(current)
            #get new flow dependinces here
            new_flow = self.find_flow_dependencies(current)

            for can in new_data:
                ok = True
                ok = ok and self.check_member(can, to_search)
                ok = ok and self.check_member(can, searched)
                if ok:
                    if self.verbose:
                        print '\tstructure', can
                    to_search.append(can)

            for can in new_flow:
                ok = True
                ok = ok and self.check_member(can, to_search)
                ok = ok and self.check_member(can, searched)
                if ok:
                    if self.verbose:
                        print '\tstructure', can
                    to_search.append(can)
            searched.add(current)



        to_print = sorted(list(searched), key=lambda x: x.distance)
        self.thresholds = []
        count = 0
        for i in to_print:
            if i in thresh:
                self.thresholds.append((i, thresh[i]))
                if self.verbose:
                    print '\n'
                    print 'Thresholds: ', thresh[i]
                    full_print(i)
                count += 1
        if self.verbose:
            print 'total thresholds {:d}'.format(count)
     


    def check_member(self, canidate, collection):
        '''check if it is a memeber and if it is
            then add it to the children  return true if
            it is not'''
        if canidate not in collection:
            return True
        else:
            if self.verbose:
                print 'Already visited:', canidate
            if self.web_style:
                col = list(collection)
                mem = col[col.index(canidate)]
                mem_calls = get_base_calls(mem)
                can_calls = get_base_calls(canidate)
                #if they are different method calls we need to combined them.
                if set(mem_calls) != set(can_calls):
                    if self.verbose:
                        print '\tdifferent calls adding to candiates'
                    for i in canidate.children:
                        if i not in mem.children:
                            if self.verbose:
                                print '\t\tAdding', mem, '<-', i
                            mem.children.append(i)
                # otherwise we need to check distances to determine what to do
                elif canidate.distance < mem.distance:
                    print "\tERROR distance violation!!!"

                elif canidate.distance == mem.distance:
                    if self.verbose:
                        print '\tsame distance'
                    for i in canidate.children:
                        if i not in mem.children:
                            mem.children.append(i)
                            if self.verbose:
                                print '\t\tAdding', mem, '<-', i

                else:
                    pass

                # for i in canidate.children:
                #         if i not in mem.children:
                #             print 'combining children'
                #             print '\t', mem.children, '<-', i
                #             mem.children.append(i)
            return False

            
    

    def find_thresholds(self, current):
        '''find any thresholds in the current statement and 
        return them if we find any'''
        #TODO: Currently only if statements
        if current.statement.node in self.if_visitor.ifs:
            return self.if_visitor.ifs[current.statement.node] 
        else:
            return [] 

    def get_vars(self, statement, node):
        vv = GetVarsVisit(statement)
        vv.visit(node)
        return vv.class_vars, vv.func_vars


    def find_data_dependiences(self, current):
        '''find any thresholds in the current statement and 
        return them if we find any'''
        to_return = []
        class_vars = set() 
        func_vars = set() 

        rd = self.reaching_defs[current.statement.cls][current.statement.func]
        if current.statement.node in rd:
            rd = rd[current.statement.node]
        else:
            if isinstance(current.statement.node, ast.Name):
                return []
            else:
                rd = rd[current.statement.expr]

        if isinstance(current.statement.node, ast.If):
            cv, fv = self.get_vars(current.statement, current.statement.node.test)
            class_vars = cv
            func_vars = fv

        elif isinstance(current.statement.node, ast.Call):
            for arg in current.statement.node.args:
                cv, fv = self.get_vars(current.statement, arg)
                for i in cv:
                    class_vars.add(i)
                for i in fv:
                    func_vars.add(i)

        elif isinstance(current.statement.node, ast.Assign):
            cv, fv = self.get_vars(current.statement, current.statement.node.value)
            for i in cv:
                class_vars.add(i)
            for i in fv:
                func_vars.add(i)

        elif isinstance(current.statement.node, ast.Expr):
            if isinstance(current.statement.node.value, ast.Call):
                for arg in current.statement.node.value.args:
                    cv, fv = self.get_vars(current.statement, arg)
                    for i in cv:
                        class_vars.add(i)
                    for i in fv:
                        func_vars.add(i)
            else:
                print 'Weird you shouldn"t be here'
        elif isinstance(current.statement.node, ast.AugAssign):
            cv, fv = self.get_vars(current.statement, current.statement.node.value)
            for i in cv:
                class_vars.add(i)
            for i in fv:
                func_vars.add(i)

        else:
            print '\nwhy are you here'
            print ast.dump(current.statement.node)
            print '\n'

        #find class statements and reachind definitions to examine next!
        for var in class_vars:
            fa = FindAssigns(var)
            fa.visit(self.tree)
            for i in fa.assignments:
                obj = SearchStruct(i, current.publisher, current, current.distance + 1)
                to_return.append(obj)

        #do function variables
        printed = False
        for fv in func_vars:
            for d in rd:
                v = fv.split('.')
                d1 = d[0].split('.')
                # print v
                # print d1
                if v == d1[:len(v)]:
                    # if not printed:
                    #     print current.statement.node.lineno, current.statement.node
                    #     printed = True
                    # print '\t->', d[1].lineno, d[0], d[1]
                    state = TreeObject(current.statement.cls, current.statement.func, d[1],d[1])
                    obj = SearchStruct(state, current.publisher, current, current.distance + 1)
                    to_return.append(obj)
                    
        return to_return


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


def full_print(obj, tabs=0, visited=None):
    print '\t' * tabs, obj
    if visited is None:
        visited = set()
    visited.add(obj)
    for child in obj.children:
        if not child in visited:
            full_print(child, tabs+1, visited)
    visited.remove(obj)

def check_important(obj, visited=None):
    if visited is None:
        visited = set()
    if obj in visited:
        return False
    if len(obj.children) == 0:
        return obj.important
    else:
        important = False
        for child in obj.children:
            important = check_important(child, visited)
        return important

def get_base_calls(thing, visited=None):
    if visited is None:
        visited = set()
    if thing in visited:
        return []
    visited.add(thing)
    values = []
    if len(thing.children) == 0:
        values.append(thing)
    else:
        for i in thing.children:
            ret = get_base_calls(i, visited) 
            for v in ret:
                values.append(v)
    return values
    


    




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
                                self.current_function, self.current_expr, node))
        self.generic_visit(node)

    


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

class AddImportStatement(ast.NodeTransformer):


    def visit_Module(self, node):
        new_node = ast.Import(names=[ast.alias(name='reporting', asname=None)])
        new_node = ast.copy_location(new_node, node.body[0])
        ast.increment_lineno(node.body[0],1)
        node.body = [new_node] + node.body
        return node

class ModCalls(ast.NodeTransformer):


    def __init__(self, ba, fname):
        self.ba = ba
        self.fname = fname
        self.tmap = {}
        for i in ba.thresholds:
            self.tmap[i[0].statement.node] = i


    def visit_If(self, node):
        if node in self.tmap:
            name = ast.Name(id='reporting', ctx=ast.Load())
            attr = ast.Attribute(value=name, attr='report', ctx=ast.Load())
            args = [node.test, ast.Str(s=str(node.lineno)), ast.Str(s=self.fname)]
            call = ast.Call(func=attr, args=args, keywords=[],starargs=None, kwargs=None)
            node.test = call 
            print 'replacing'
        return node





def replace_values(tree, back_analysis, fname):

    tree = ModCalls(back_analysis, fname).visit(tree)
    tree = AddImportStatement().visit(tree)
    ast.fix_missing_locations(tree)

    code =compile(tree,fname ,mode='exec')
    print dir(code)
    ns = {'__name__' : '__main__'}
    exec(code, ns)


def analyze_file(fname, execute=False):
    '''new main function...get CFG and find pubs first'''
    print '\n\n', fname, ':'
    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  

            a = AssignFindVisitor()
            a.visit(tree)
            canidates = CanidateStore(a.canidates, tree)

            flow_store = cfg_analysis.build_files_cfgs(tree=tree)

            rd = ReachingDefinition(tree, flow_store)
            rd.compute()

            publish_finder = PublishFinderVisitor()
            publish_finder.visit(tree)
            calls = publish_finder.publish_calls
            ba = BackwardAnalysis(canidates, calls, flow_store, tree, rd.rds_in, False, True)
            ba.compute()
            print len(ba.thresholds)
            for i in ba.thresholds:
                full_print(i[0])

            if execute:
                print 'working on execution'
                tree = replace_values(tree, ba, fname) 
            

            # for i in rd.rds_in:
            #     keys =rd.rds_in[i]
            #     for key, values in keys.iteritems():
            #         print key.lineno, key
            #         vals = sorted(values.keys(), key=lambda x: x.lineno)
            #         for i in vals:
            #             print '\t', i.lineno, i, '->'
            #             for k in values[i]:
            #                 print '\t\t', k[0], k[1].lineno
            #             print 


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
    analyze_file(args.file, True)
