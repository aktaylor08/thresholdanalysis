#!/usr/bin/env python
# encoding: utf-8
from __future__ import print_function

import ast
import os
import argparse
import pprinter
import sys

import cfg_analysis

from collections import defaultdict, deque
from ast_tools import get_name, get_string_repr, get_node_code


class TreeObject(object):

    """'hold all of the information needed
    about a cfg node in this stuff"""

    def __init__(self, cls, func, expr, node):
        self.cls = cls
        self.func = func
        self.expr = expr
        self.node = node

    def get_repr(self, code):
        return str(self.node.lineno) + ': ' + get_node_code(self.node, code)

    def get_full_dict(self, code):
        ret = {'cls': get_node_code(self.cls, code), 'func': get_node_code(self.func, code),
               'expr': get_node_code(self.expr, code), 'node': get_node_code(self.node, code)}
        return ret

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.node.lineno) + ': ' + str(self.node)

    def __eq__(self, other):
        cls = self.cls == other.cls
        func = self.func == other.func
        node = self.node == other.node
        expr = self.expr = other.expr
        return cls and func and node and expr

    def __hash__(self):
        return hash(self.cls) + hash(self.func) + hash(self.node) + hash(self.expr)

    @staticmethod
    def print_repr(code):
        print(get_string_repr(code))


class ClassVariable(object):

    """holds information about a class variable"""

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

    """holds information about a Function variable"""

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

    """data structure that holds information for the search/backward
    analysis"""

    def __init__(self, statement, publisher, children, distance, important=False):
        self.statement = statement
        self.publisher = publisher
        if children is None:
            self.children = []
        elif not isinstance(children, list):
            self.children = [children]
        else:
            self.children = children
        self.distance = distance
        self.parent = None
        self.important = important

    def get_repr(self, code):
        return str(self.distance) + ' ' + self.statement.get_repr(code)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.distance) + ' ' + str(self.statement)

    def __eq__(self, other):
        return self.statement.node == other.statement.node

    def __hash__(self):
        return hash(self.statement.node)

    @staticmethod
    def print_repr(code):
        print(get_string_repr(code))


class CanidateStore(object):

    """class to hold all of the candidates for
        thresholds.  Will include num literals, class variables,
        and variables within a function"""

    def __init__(self, assignments, tree):
        """build the list from found assignments"""
        self.assignments = assignments
        self.tree = tree
        self.class_vars = {}
        self.func_vars = {}
        self.compile_canidates()

    def compile_canidates(self):
        """compile all of the assignments down into a list that we can check
        to see what they really assign and how many times they are assigned"""
        self.do_class_variables()
        self.do_func_variables()

    def do_class_variables(self):
        """Define all of the class variables as constants or not"""
        # book keeping
        for_certain = set()
        bad = set()
        init = set()
        elsewhere = set()
        maybe = set()
        classes = self.assignments.keys()

        for cls in classes:
            variables = sorted(
                self.assignments[cls], key=lambda x: x.func.name)
            for i in variables:
                if isinstance(i, ClassVariable):
                    if isinstance(i.assign, ast.AugAssign):
                        bad.add(i)
                    else:
                        # if it makes a call to rospy.get_param it is a
                        # threshold
                        if isinstance(i.assign.value, ast.Call):
                            if self.is_paramcall(i.assign.value):
                                for_certain.add(i)

                        # as of right now we just increment in init but
                        if i.func.name == '__init__':
                            init.add(i)
                        else:
                            if self.check_only_const(i):
                                maybe.add(i)
                            else:
                                elsewhere.add(i)

        vals = init.union(maybe).difference(
            elsewhere).difference(bad).union(for_certain)
        for i in vals:
            if i.cls in self.class_vars:
                self.class_vars[i.cls].append(i)
            else:
                self.class_vars[i.cls] = [i]

    def do_func_variables(self):
        """Define function variables as constants or not"""
        classes = self.assignments.keys()
        candidates = {}
        bad = set()
        for cls in classes:
            variables = sorted(
                self.assignments[cls], key=lambda sort_val: sort_val.name)
            for i in variables:
                if isinstance(i, FunctionVariable):
                    if isinstance(i.assign, ast.AugAssign):
                        bad.add(i)
                    else:
                        const = self.check_only_const(i.assign.value)
                        if const:
                            if i in candidates:
                                candidates[i] += 1
                            else:
                                candidates[i] = 1
                        else:
                            bad.add(i)
        vals = [x for x in candidates.keys() if candidates[x] == 1]
        vals = set(vals)
        vals = vals - bad
        for i in vals:
            if i.cls in self.func_vars:
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
            # check to see if it is a call to what is a constant param
            return self.is_paramcall(node)
        elif isinstance(node, ast.Attribute):
            return self.is_const(node)

    @staticmethod
    def is_paramcall(node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'rospy' and node.func.attr == 'get_param':
                    return True
        return False

    def is_const(self, node):
        """given a node of an ast return if it is
        a constant candidate -> looks up name and  other stuff"""
        cls, func = self.get_class_and_func(node)

        # if it is an attribute than check if it is a self call and
        # then check to see if it is in its class listing
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


def get_gen(node):
    """gen set -> any assignment"""
    to_return = set()
    if isinstance(node, ast.Assign):
        for target in node.targets:
            to_return.add((get_name(target), node))
    elif isinstance(node, ast.AugAssign):
        target = node.target
        to_return.add((get_name(target), node))
    return to_return


class ReachingDefinition(object):

    """class to compute reaching definitions on all functions within
    a file.  Will compute both exit and enter values for all of them"""

    def __init__(self, tree, cfg_store):
        """build"""
        self.tree = tree
        self.cfg_store = cfg_store
        self.rds_in = {}
        self.rds_out = {}

    def compute(self):
        """compute RD for each function"""
        for i in self.cfg_store:
            self.rds_out[i] = {}
            self.rds_in[i] = {}
            for func in self.cfg_store[i]:
                ins, outs = self.do_function(self.cfg_store[i][func])
                self.rds_out[i][func] = outs
                self.rds_in[i][func] = ins

    def do_function(self, cfg):
        """compute ins and outs for a function
        start off with any params that are not self in the function
        and than do some iteration until you reach a fix point"""
        outs = {}
        ins = {}
        for i in cfg.preds:
            outs[i] = set()
        func = list(cfg.preds[cfg.start])[0]
        # handle the arguments
        arguments = func.args.args
        for arg in arguments:
            if isinstance(arg, ast.Name):
                if arg.id == 'self':
                    pass
                else:
                    outs[func].add((arg.id, arg))
            else:
                print('ERROR line 325ish', file=sys.stderr)

        # now we will iterate until something changes
        changed = True
        while changed:
            seen = set()
            node = cfg.start
            changed = self.iterate(seen, node, outs, cfg)

        # ins are just the union of the preceding outs.
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
        """this is the main function that compute gens
        and kills and than does the union on the entering data"""
        if node in seen:
            return False
        if isinstance(node, ast.FunctionDef):
            return False

        changed = False
        # add initials
        values = cfg.preds[node]
        ins = set()
        for val in values:
            for to_add in outs[val]:
                ins.add(to_add)

        # gen kill set operations
        gen = get_gen(node)
        kill = self.get_kill(node, ins)
        temp = (ins - kill)
        for one_gen in gen:
            temp.add(one_gen)
        if temp != outs[node]:
            changed = True
            outs[node] = temp

        # keep track
        seen.add(node)
        # visit all the successors
        if node in cfg.succs:
            for i in cfg.succs[node]:
                changed = self.iterate(seen, i, outs, cfg) or changed
        return changed

    @staticmethod
    def get_kill(node, current):
        """kill set -> here its any assignment
        :param node:
        :param current:
        """
        to_return = set()
        name = None
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


class AssignPrinter(ast.NodeVisitor):
    """Class to print out assignments"""

    def __init__(self, src_code):
        self.code = src_code

    def visit_Assign(self, node):
        print('\t', node.lineno, get_node_code(node, self.code))

    def visit_AugAssign(self, node):
        print('\t', node.lineno, get_node_code(node, self.code))


class PrintIfVisitor(ast.NodeVisitor):

    """Super simple visitor to print out all encountered if functions"""

    def __init__(self, src_code):
        self.code = src_code

    def visit_If(self, node):
        print('\t', get_node_code(node, self.code))


class BasicVisitor(ast.NodeVisitor):

    """this is a super simple visitor
    which keeps track of the current class
    and function that you are in while traversing
    the tree.  Can be extended to keep the functionality
    without having to copy a bunch of code"""

    def __init__(self):
        """start the tracking"""
        self.current_class = None
        self.current_function = None
        self.current_expr = None

        self._flevel = 0

    def visit_Module(self, node):
        """we set the module level as the current class"""
        self.current_class = node
        self.current_function = ast.FunctionDef('GLOBAL_FUNCTION', [], [], [])
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        """do some assignments"""
        old_func = self.current_function
        self.current_function = node
        self.generic_visit(node)
        self.current_function = old_func

    def visit_ClassDef(self, node):
        """do some more assignments"""
        old_class = self.current_class
        self.current_class = node
        self.generic_visit(node)
        self.current_class = old_class

    def visit_Expr(self, node):
        self.current_expr = node
        self.generic_visit(node)
        self.current_expr = None

    def visit_Assign(self, node):
        self.current_expr = node
        self.generic_visit(node)
        self.current_expr = None

    def visit_Return(self, node):
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


class AssignFindVisitor(BasicVisitor):

    """find all of the assignments and organize them
    into global and class lists"""

    def __init__(self, src_code):
        """save symbol table and current class and
        locations"""
        BasicVisitor.__init__(self)
        self.src_code = src_code
        self.canidates = defaultdict(list)

    def handle_attribute(self, attr, node):
        name = get_name(attr)
        if name.startswith('self.'):
            # class value save it here
            self.canidates[self.current_class].append(ClassVariable(
                self.current_class, self.current_function, attr.attr, node))
        else:
            self.canidates[self.current_class].append(FunctionVariable(
                self.current_class, self.current_function, get_name(attr), node))

    def handle_name(self, name, node):
        self.canidates[self.current_class].append(FunctionVariable(
            self.current_class, self.current_function, name.id, node))

    def handle_subscript(self, sub, node):
        if isinstance(sub.value, ast.Attribute):
            self.handle_attribute(sub.value, node)
        elif isinstance(sub.value, ast.Name):
            self.handle_name(sub.value, node)

    def visit_assign_things(self, queue, node):
        for i in queue:
            # assigning to self.value is an attribute
            if isinstance(i, ast.Attribute):
                self.handle_attribute(i, node)
            elif isinstance(i, ast.Name):
                self.handle_name(i, node)
            elif isinstance(i, ast.Subscript):
                self.handle_subscript(i, node)
            elif isinstance(i, ast.List):
                for elt in i.elts:
                    queue.append(elt)
            else:
                print('\nERROR unimplemented AST Type:',
                      node.lineno, type(i), file=sys.stderr)
                print(get_node_code(node, self.src_code), file=sys.stderr)

    def get_tuple_elements(self, tup):
        vals = []
        for i in tup.elts:
            if isinstance(i, ast.Tuple):
                v = self.get_tuple_elements(i)
                for x in v:
                    vals.append(x)
            else:
                vals.append(i)
        return vals

    def visit_Assign(self, node):
        """visit an assignment definition"""

        # we are going to look at all of the assign values here and figure out
        # if it is a constant.  Here we are just looking at __init__ for now but
        # it could be in many other location
        queue = []
        for i in node.targets:
            if isinstance(i, ast.Tuple):
                vals = self.get_tuple_elements(i)
                for x in vals:
                    queue.append(x)
            else:
                queue.append(i)
        self.visit_assign_things(queue, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        """visit augmented assignments"""
        queue = []
        if isinstance(node.target, ast.Tuple):
            vals = self.get_tuple_elements(node.target)
            for x in vals:
                queue.append(x)
        else:
            queue = [node.target]
        self.visit_assign_things(queue, node)
        self.generic_visit(node)


class IfOrFuncVisitor(BasicVisitor):

    """finds if the program is part of an if statement"""

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
                    self.res = TreeObject(self.current_class, self.current_function,
                                          self.current_expr, temp)
                    break
            # WTF?
            for _ in reversed(popped):
                self.canidates.append(popped)
        else:
            BasicVisitor.generic_visit(self, node)


class ServiceFinderVisitor(BasicVisitor):

    def __init__(self):
        BasicVisitor.__init__(self)
        self.proxies = []

    def visit_Assign(self, node):

        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                if isinstance(node.value.func.value, ast.Name):
                    if node.value.func.value.id == 'rospy' and node.value.func.attr == 'ServiceProxy':
                        for i in node.targets:
                            name = get_name(i)
                            if name.startswith('self'):
                                cv = ClassVariable(
                                    self.current_class, self.current_function, name, node)
                                self.proxies.append(cv)
                            else:
                                fv = FunctionVariable(
                                    self.current_class, self.current_function, name, node)
                                self.proxies.append(fv)


class ServiceCallFinder(BasicVisitor):

    def __init__(self, proxies):
        BasicVisitor.__init__(self)
        self.proxies = proxies
        self.calls = []

    def visit_Call(self, node):
        func = node.func
        name = get_name(func)
        if name.startswith('self'):
            # check function variable:w
            cv = ClassVariable(
                self.current_class, self.current_function, name, node)
            if cv in self.proxies:
                self.calls.append(
                    TreeObject(self.current_class,
                               self.current_function, self.current_expr, node))

        else:
            fv = FunctionVariable(
                self.current_class, self.current_function, name, node)
            if fv in self.proxies:
                if self.current_expr is None:
                    print(node, file=sys.stderr)
                    print(node.lineno, file=sys.stderr)
                    print(pprinter.dump(node), file=sys.stderr)
                    print('\n\n', file=sys.stderr)
                self.calls.append(
                    TreeObject(self.current_class,
                               self.current_function, self.current_expr, node))


class PublishFinderVisitor(BasicVisitor):

    """find and store all of the rospy.publish calls
    in this manner we can get all of the functions and
    stuff that they reside in  will store them in an object"""

    def __init__(self):
        BasicVisitor.__init__(self)
        self.publish_calls = []

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Name):
            # skipping for now
            pass
        elif isinstance(func, ast.Attribute):
            if func.attr == 'publish':
                self.publish_calls.append(
                    TreeObject(self.current_class,
                               self.current_function, self.current_expr, node))


def get_local_pub_srv(tree):
    publish_finder = PublishFinderVisitor()
    publish_finder.visit(tree)
    service_finder = ServiceFinderVisitor()
    service_finder.visit(tree)
    call_finder = ServiceCallFinder(service_finder.proxies)
    call_finder.visit(tree)
    return call_finder.calls + publish_finder.publish_calls


class InterestingStatementStore(object):

    def __init__(self, tree=None, src_code=None):
        self.tree = tree
        self.src_code = src_code
        self.calls = list()
        self.get_local_pub_srv(tree)
        self.find_import_values()

    def find_import_values(self):
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                print(node)
            elif isinstance(node, ast.ImportFrom):
                print(node)


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
        """add it to the function variables"""
        self.func_vars.add(node.id)

    def visit_Attribute(self, node):
        """if it is an attribute check the name
        to see if it starts with self"""
        name = get_name(node)
        self.func_vars.add(name)
        if name.startswith('self.'):
            # we have a class variable so check it out
            cv = ClassVariable(self.statement.cls, self.statement.func,
                               name, node)
            self.class_vars.add(cv)


class FindAssigns(BasicVisitor):

    def __init__(self, var):
        BasicVisitor.__init__(self)
        self.var = var
        self.assignments = []

    def visit_Assign(self, node):
        """visit an assignment definition"""

        # we are going to look at all of the assign values here and figure out
        # if it is a constant.  Here we are just looking at __init__ for now but
        # it could be in many other location
        for i in node.targets:
            # assigning to self.value is an attribute
            name = get_name(i)
            names = name.split('.')
            varsplit = self.var.name.split('.')
            if name.startswith('self.') and names == varsplit[:len(names)]:
                if self.current_function != self.var.func:
                    self.assignments.append(TreeObject(self.current_class,
                                                       self.current_function, self.current_expr, node))

    def visit_AugAssign(self, node):
        i = node.target
        # assigning to self.value is an attribute
        name = get_name(i)
        names = name.split('.')
        varsplit = self.var.name.split('.')
        if name.startswith('self.') and names == varsplit[:len(names)]:
            if self.current_function != self.var.func:
                self.assignments.append(TreeObject(self.current_class,
                                                   self.current_function, self.current_expr, node))


class BackwardAnalysis(object):

    """class to perform the backward analysis needed on all of the files"""

    def __init__(self, if_visitor, calls, flow_store, tree, reaching_defs, verbose=False, web=False, src_code=None):
        self.calls = calls
        self.flow_store = flow_store
        self.tree = tree
        self.if_visitor = if_visitor
        self.reaching_defs = reaching_defs
        self.verbose = verbose
        self.web_style = web
        self.thresholds = []
        self.src_code = src_code

    def compute(self):
        searched = set()
        to_search = deque()
        thresh = {}
        # add important statements
        for call in self.calls:
            obj = SearchStruct(call, call, None, 0, important=True)
            to_search.append(obj)
        while len(to_search) > 0:
            current = to_search.popleft()
            if self.verbose:
                print('\n')
                print(current.get_repr(self.src_code))
            # find some thresholds
            new_thresholds = self.find_thresholds(current)
            if len(new_thresholds) > 0:
                thresh[current] = new_thresholds
                if self.verbose:
                    print('\tFOUND THRESHOLD!:')
                for i in new_thresholds:
                    pass
                    if self.verbose:
                        print(i)
            # get data flows from here
            new_data = self.find_data_dependiences(current)
            # get new flow dependence here
            new_flow = self.find_flow_dependencies(current)

            for can in new_data:
                ok = True
                ok = ok and self.check_member(can, to_search)
                ok = ok and self.check_member(can, searched)
                if ok:
                    if self.verbose:
                        print('\tdata:', can.get_repr(self.src_code))
                    to_search.append(can)

            for can in new_flow:
                ok = True
                ok = ok and self.check_member(can, to_search)
                ok = ok and self.check_member(can, searched)
                if ok:
                    if self.verbose:
                        print('\tstructure:', can.get_repr(self.src_code))
                    to_search.append(can)
            searched.add(current)

        to_print = sorted(list(searched), key=lambda x: x.distance)
        self.thresholds = []
        count = 0
        for i in to_print:
            if i in thresh:
                self.thresholds.append((i, thresh[i]))
                if self.verbose:
                    print('\n')
                    print('Thresholds: ', thresh[i])
                    full_print(i)
                count += 1
        if self.verbose:
            print('total thresholds {:d}'.format(count))

    def check_member(self, canidate, collection):
        """check if it is a member and if it is
            then add it to the children  return true if
            it is not"""
        if canidate not in collection:
            return True
        else:
            if self.verbose:
                print('Already visited:', canidate)
            if self.web_style:
                col = list(collection)
                mem = col[col.index(canidate)]
                mem_calls = get_base_calls(mem)
                can_calls = get_base_calls(canidate)
                # if they are different method calls we need to combined them.
                if set(mem_calls) != set(can_calls):
                    if self.verbose:
                        print('\tdifferent calls adding to candidates')
                    for i in canidate.children:
                        if i not in mem.children:
                            if self.verbose:
                                print('\t\tAdding', mem, '<-', i)
                            mem.children.append(i)
                # otherwise we need to check distances to determine what to do
                elif canidate.distance < mem.distance:
                    print("\tERROR distance violation!!!")

                elif canidate.distance == mem.distance:
                    if self.verbose:
                        print('\tsame distance')
                    for i in canidate.children:
                        if i not in mem.children:
                            mem.children.append(i)
                            if self.verbose:
                                print('\t\tAdding', mem, '<-', i)

                else:
                    pass

                    # for i in canidate.children:
                    # if i not in mem.children:
                    # print 'combining children'
                    # print '\t', mem.children, '<-', i
                    # mem.children.append(i)
            return False

    def find_thresholds(self, current):
        """find any thresholds in the current statement and
        return them if we find any"""
        # TODO: Currently only if statements
        if current.statement.node in self.if_visitor.ifs:
            return self.if_visitor.ifs[current.statement.node]
        else:
            return []

    @staticmethod
    def get_vars(statement, node):
        vv = GetVarsVisit(statement)
        vv.visit(node)
        return vv.class_vars, vv.func_vars

    def find_data_dependiences(self, current):
        """find any thresholds in the current statement and
        return them if we find any"""
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
                try:
                    rd = rd[current.statement.expr]
                except KeyError:
                    print(
                        current.statement.get_repr(self.src_code), file=sys.stderr)
                    print(current.get_repr(self.src_code), file=sys.stderr)
                    print(current.statement.expr, file=sys.stderr)
                    print('reaching definition exceptions', file=sys.stderr)
                    full_print(current)
                    assert False

        if isinstance(current.statement.node, ast.If):
            cv, fv = self.get_vars(
                current.statement, current.statement.node.test)
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
            cv, fv = self.get_vars(
                current.statement, current.statement.node.value)
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
                print('Weird you should not be here', file=sys.stderr)
        elif isinstance(current.statement.node, ast.AugAssign):
            cv, fv = self.get_vars(
                current.statement, current.statement.node.value)
            for i in cv:
                class_vars.add(i)
            for i in fv:
                func_vars.add(i)

        else:
            print('\nwhy are you here', file=sys.stderr)
            print(ast.dump(current.statement.node), file=sys.stderr)
            print('\n', file=sys.stderr)

        # find class statements and reaching definitions to examine next!
        for var in class_vars:
            fa = FindAssigns(var)
            fa.visit(self.tree)
            for i in fa.assignments:
                obj = SearchStruct(
                    i, current.publisher, current, current.distance + 1)
                to_return.append(obj)

        # do function variables
        for fv in func_vars:
            for d in rd:
                v = fv.split('.')
                d1 = d[0].split('.')
                if v == d1[:len(v)]:
                    state = TreeObject(
                        current.statement.cls, current.statement.func, d[1], d[1])
                    obj = SearchStruct(
                        state, current.publisher, current, current.distance + 1)
                    to_return.append(obj)

        return to_return

    def find_flow_dependencies(self, current):
        """find flow dependencies"""
        visitor = IfOrFuncVisitor(current.statement.node)
        visitor.visit(self.tree)
        to_return = []
        if isinstance(visitor.res.node, ast.If):
            obj = SearchStruct(
                visitor.res, current.publisher, current, current.distance + 1)
            to_return.append(obj)
        else:
            # otherwise search for function calls here?
            for call in self.search_function_calls(visitor.res):
                obj = SearchStruct(
                    call, current.publisher, current, current.distance + 1)
                to_return.append(obj)

        return to_return

    def search_function_calls(self, tree_thing):
        fcv = FindCallVisitor(tree_thing.cls, tree_thing.func)
        fcv.visit(self.tree)
        return fcv.calls


def full_print(obj, tabs=0, visited=None, code=None):
    if code is not None:
        print('\t' * tabs, obj.get_repr(code))
    else:
        print('\t' * tabs, obj)
    if visited is None:
        visited = set()
    visited.add(obj)

    for child in obj.children:
        if child not in visited:
            full_print(child, tabs + 1, visited, code)
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


class FindCallVisitor(BasicVisitor):

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
                        # save this for use
                        self.calls.append(TreeObject(self.current_class,
                                                     self.current_function, self.current_expr, node))
        self.generic_visit(node)


class IfConstantVisitor(BasicVisitor):

    """visit if statements to ID which constants are
    used in if statements"""

    def __init__(self, canidates):
        BasicVisitor.__init__(self)
        self.canidates = canidates
        self.ifs = {}

    def visit_If(self, node):
        cv = ConstantVisitor(self.canidates, self.current_class,
                             self.current_function)
        cv.visit(node.test)
        if len(cv.consts) > 0:
            self.ifs[node] = cv.consts
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

    """IDs constants from candidates and also numerical constants"""

    def __init__(self, canidates, cls, func):
        BasicVisitor.__init__(self)
        self.canidates = canidates
        self.consts = []
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
        fv = FunctionVariable(self.current_class,
                              self.current_function, node.id, node)
        if self.current_class in self.canidates.func_vars:
            if fv in self.canidates.func_vars[self.current_class]:
                self.consts.append(fv)


def add_import_statement(node):
    new_node = ast.Import(names=[ast.alias(name='reporting', asname=None)])
    new_node = ast.copy_location(new_node, node.body[0])
    ast.increment_lineno(node.body[0], 1)
    node.body = [new_node] + node.body
    return node


class ModCalls(ast.NodeTransformer):

    def __init__(self, ba, fname, code, verbose):
        self.ba = ba
        self.fname = fname
        self.tmap = {}
        self.code = code
        self.verbose = verbose
        for i in ba.thresholds:
            self.tmap[i[0].statement.node] = i

    def visit_If(self, node):
        if node in self.tmap:
            if self.verbose:
                print('modifying:', node.lineno, node)
            line_code = self.code[node.lineno - 1].lstrip().strip()
            nav = NameAttrVisitor(self.fname)
            nav.visit(node.test)
            for i in nav.things:
                ast.fix_missing_locations(i.value)

            name = ast.Name(id='reporting', ctx=ast.Load())
            attr = ast.Attribute(value=name, attr='report', ctx=ast.Load())
            func_args = [node.test, ast.Str(s=self.fname), ast.Str(
                s=str(node.lineno)), ast.Str(s=line_code)]

            # print nav.things
            # nav.things)
            call = ast.Call(
                func=attr, args=func_args, keywords=nav.things, starargs=None, kwargs=None)
            node.test = call
        self.generic_visit(node)
        return node


class NameAttrVisitor(ast.NodeVisitor):

    def __init__(self, name_pre):
        self.things = []
        self.name_pre = name_pre

    def visit(self, node):
        # print node
        ast.NodeVisitor.visit(self, node)

    # AT the moment I'm skipping expressions. Not sure if they need to visited
    # or not
    def visit_UnaryOp(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)

        self.visit(node.operand)

    def visit_BinOp(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)

        self.visit(node.left)
        self.visit(node.right)

    def visit_BoolOp(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)

        for i in node.values:
            self.visit(i)

    def visit_Compare(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)

        self.visit(node.left)
        for i in node.comparators:
            self.visit(i)

    def visit_Call(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)

        for i in node.args:
            self.visit(i)

        for i in node.keywords:
            self.visit(i)
            # TODO Ignoring starargs and kwargs for now

    def visit_Attribute(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)

    def visit_Name(self, node):
        name = self.name_pre + \
            str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)


def replace_values(tree, back_analysis, fname, code, verbose):
    tree = ModCalls(back_analysis, fname, code, verbose).visit(tree)
    tree = add_import_statement(tree)
    ast.fix_missing_locations(tree)

    code = compile(tree, fname, mode='exec')
    ns = {'__name__': '__main__'}
    exec (code, ns)


def close_graph(node, graph, visited):
    if node in visited:
        return
    if node not in graph:
        return
    visited.add(node)
    for target in graph[node]:
        close_graph(target, graph, visited)


def get_code(fname):
    """Get the code from a file"""
    if os.path.isfile(fname):
        with open(fname, 'r') as openf:
            code = openf.read()
            spcode = code.split('\n')
            return code, spcode
    else:
        print('error no file')
        return None, None


def get_code_and_tree(fname):
    """Return the source code list and the ast tree"""
    code, spcode = get_code(fname)
    tree = get_tree(code)
    return spcode, tree


def get_tree(code):
    """Return a tree from the code passed in here"""
    tree = ast.parse(code)
    return tree


def get_candidates(tree, src_code, verbose=False):
    if verbose:
        print('finding assignments')
    a = AssignFindVisitor(src_code)
    a.visit(tree)
    if verbose:
        print('done finding assignments')

    if verbose:
        print('Pruning to canidate set')
    canidates = CanidateStore(a.canidates, tree)
    return canidates


def get_cfg(tree, src_code, verbose=False):
    if verbose:
        print('Building control flow graph')
    flow_store = cfg_analysis.build_files_cfgs(
        tree=tree, verbose=verbose, src_code=src_code)
    return flow_store


def get_reaching_definitions(tree, flow_store, verbose=False):
    if verbose:
        print('Computing reaching definition')
    rd = ReachingDefinition(tree, flow_store)
    rd.compute()
    return rd


def get_pub_srv_calls(tree, src_code, verbose=False):

    if verbose:
        print('Finding interesting calls')
    iss = InterestingStatementStore(tree, src_code)
    calls = iss.calls
    if verbose:
        print('\nPub and service calls: ')
        for i in calls:
            print('\t', i.get_repr(src_code))
    return calls


def get_const_ifs(candidates, tree, spcode, verbose=False):
    if verbose:
        print('\nFinding if statements with constant values')
    if_visit = IfConstantVisitor(candidates)
    if_visit.visit(tree)
    if verbose:
        print("Following if statements have constants: ")
        for i in if_visit.ifs:
            print(get_node_code(i, spcode))
    return if_visit


def perform_analysis(if_visit, calls, flow_store, tree, rd, verbose=False, web=False, src_code=None):
    if verbose:
        print('\nfinding thresholds in file')
    ba = BackwardAnalysis(if_visit, calls, flow_store, tree,
                          rd.rds_in, verbose=verbose, web=web, src_code=src_code)
    ba.compute()
    if verbose:
        for i in ba.thresholds:
            full_print(i[0], code=src_code)
            print("\n")
    return ba


def analyze_file(fname, verbose=False, execute=False):
    """new main function...get CFG and find pubs first"""
    if os.path.isfile(fname):
        print('File: ', fname)
        if verbose:
            print('parsing file')
        src_code, tree = get_code_and_tree(fname)
        candidates = get_candidates(tree,  src_code, verbose)
        flow_store = get_cfg(tree, src_code, verbose)
        rd = get_reaching_definitions(tree, flow_store, verbose)
        calls = get_pub_srv_calls(tree,  src_code, verbose)
        if_visit = get_const_ifs(candidates, tree,  src_code, verbose)
        ba = perform_analysis(if_visit, calls, flow_store, tree, rd,
                              verbose=verbose, web=False, src_code=src_code)

        print('Number of Thresholds:', len(ba.thresholds))
        for i in ba.thresholds:
            print(i[0], i[1])

        if execute:
            print('\nnow modifying source code\n')
            replace_values(tree, ba, fname, src_code, verbose)

    else:
        print('error no file')


def list_ifs(fname):
    """Quickly print a list of all the if statements in the file"""
    src_code, tree = get_code_and_tree(fname)
    print('\nIf statements in {:s}: '.format(fname))
    PrintIfVisitor(src_code).visit(tree)


def list_assigns(fname):
    src_code, tree = get_code_and_tree(fname)
    print('Assignments in {:s}'.format(fname))
    AssignPrinter(src_code).visit(tree)


def list_constants(fname):
    src_code, tree = get_code_and_tree(fname)
    candidates = get_candidates(tree, src_code)
    print("\nIdentified Constants in {:s}".format(fname))
    for cls in candidates.class_vars:
        print('\tClass: {:s}'.format(cls.name))
        for cls_var in candidates.class_vars[cls]:
            print('\t\t{:s}'.format(cls_var))

    print('\tFunction Constants:')
    for cls in candidates.func_vars:
        for func_var in candidates.func_vars[cls]:
            print('\t\t{:s}'.format(func_var))


def list_pubs(fname):
    src_code, tree = get_code_and_tree(fname)
    print('\nPublish Calls in {:s}'.format(fname))
    for pub in get_pub_srv_calls(tree, src_code):
        print('\n\t'.join([k + ' : ' + v for k, v in pub.get_full_dict(src_code).iteritems()]))


def list_constant_ifs(fname):
    src_code, tree = get_code_and_tree(fname)
    candidates = get_candidates(tree, src_code)
    if_visit = get_const_ifs(candidates, tree, src_code)
    print('\nIf statements with Constant Values in {:s}: '.format(fname))
    for i in if_visit.ifs:
        print('\t', i.lineno, '->', get_node_code(i, src_code))


def list_cfg(fname):
    src_code, tree = get_code_and_tree(fname)
    get_cfg(tree, src_code=src_code, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("This is a program to find"
                                                  " constant thresholds in a python program"))
    parser.add_argument('file', help='path to file')
    parser.add_argument('-n', '--no_execute', help='Set execution to false',
                        action='store_true', )
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        action='store_true', )
    parser.add_argument('rest', nargs='*')
    parser.add_argument(
        '--list-ifs', help='List all if statements and exit', action='store_true')
    parser.add_argument(
        '--list-const-ifs', help='List all if statements that contain constants', action='store_true')
    parser.add_argument(
        '--list-constants', help='List all of the identified constants', action='store_true')
    parser.add_argument(
        '--list-pubs', help='List all of the statements IDed as publishers', action='store_true')
    parser.add_argument(
        '--list-assign', help='List all of the assignments in code', action='store_true')
    parser.add_argument(
        '--list-cfg', help='Print the CFG created for the file', action='store_true')
    args = parser.parse_args()
    no_analysis = False
    if args.list_ifs:
        list_ifs(args.file)
        no_analysis = True
    if args.list_constants:
        list_constants(args.file)
        no_analysis = True
    if args.list_const_ifs:
        list_constant_ifs(args.file)
        no_analysis = True
    if args.list_pubs:
        list_pubs(args.file)
        no_analysis = True
    if args.list_assign:
        list_assigns(args.file)
        no_analysis = True
    if args.list_cfg:
        list_cfg(args.file)
        no_analysis = True

    if not no_analysis:
        analyze_file(
            args.file, verbose=args.verbose, execute=not args.no_execute)
