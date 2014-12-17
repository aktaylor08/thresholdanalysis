#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import argparse

import ast_tools


class FunctionCFGStore(object):
    """Store for the CFG information instead of the actual visitor itself"""

    def __init__(self):
        self.succs = dict()
        self.preds = dict()
        self.start = None
        self.last = None


class CFGVisitor(ast.NodeVisitor):

    """build the CFG for a function in python"""

    def __init__(self, func):
        """initilize the function"""
        self._statements = set()
        self._init_map = dict()
        self._func = func
        self._start = None
        self._last = func
        self._succs = dict()
        self._preds = dict()
        self._current_loop = []
        self._next_loop_target = []
        self._try_targets = []
        self.stores = dict()


    def visit_FunctionDef(self, node):
        if node != self._func:
            sub_graph_visitor = CFGVisitor(node)
            sub_graph_visitor.start_visit(node)
            for k, v in sub_graph_visitor.stores.iteritems():
                self.stores[k] = v
        self.generic_visit(node)

    def start_visit(self, node):
        """ here lies the start of it all.  Add some bookkeeping edges"""
        # set the start to be the first node in body
        self._start = node.body[0]
        # the last one is the function itself
        # now start the cfg by adding one to the end
        self._last = node

        # one node body is also different
        self.add_edge(node.body[-1], self._last)
        self.handleBlock(node.body)
        self.generic_visit(node)
        self.buildCFG(set(), self._start)
        # add the start node to the pred graph so we have a close one
        self._preds[self._start] = set([node])
        store = FunctionCFGStore()
        store.succs = self._succs
        store.preds = self._preds
        store.start = self._start
        store.last = self._last
        self.stores[node] = store



    def visit_While(self, node):
        """pass it to the loop handler"""
        self.handle_loop(node)

    def visit_For(self, node):
        """pass it to the loop handler"""
        self.handle_loop(node)

    def visit_TryFinally(self, node):
        """visit a try finally block"""
        # get where this node points
        nnode = self.get_target(node)
        # clear the nodes target and point it at the body
        self._init_map[node].clear()

        # point to the correct things
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node.finalbody[0])
        self.add_edge(node.finalbody[-1], nnode)

        # handle both blocks and continue on
        self.handleBlock(node.body)
        self.handleBlock(node.finalbody)
        self.generic_visit(node)

    def visit_TryExcept(self, node):
        """try excpts"""
        target = self.get_target(node)
        # point to some of the right things
        self._init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], target)

        if len(node.handlers) == 1:
            # if only once except than point that one to the target
            self.add_edge(node.handlers[0], target)
        else:
            # otherwise add all of the nodes and  stuff
            first = node.handlers[0]
            self.add_edge(first, target)
            for nh in node.handlers[1:]:
                self.add_edge(first, nh)
                self.add_edge(nh, target)
                first = nh

        # add exception handlers to be pointed at only point to first one
        self._try_targets.append(node.handlers[0])

        # visit
        self.handleBlock(node.body)
        self.generic_visit(node)
        # remove them
        self._try_targets = self._try_targets[:-1]

    def visit_ExceptHandler(self, node):
        """here we visit an exception handler"""
        target = self.get_target(node)

        # we are good so
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], target)
        # pop one off before we visit some blocks because they will be
        # already visited and don't want to point to ourselves
        tt = self._try_targets[-1]
        self._try_targets = self._try_targets[:-1]
        self.handleBlock(node.body)
        self._try_targets.append(tt)
        self.generic_visit(node)

    def visit_If(self, node):
        """visit an if node"""
        nnode = self.get_target(node)
        self._init_map[node].clear()
        # do stuff with the then list
        then_list = node.body
        self.add_edge(node, then_list[0])
        for eh in self._try_targets:
            self.add_edge(node, eh)
        self.add_edge(then_list[-1], nnode)
        # handle try targets
        for eh in self._try_targets:
            self.add_edge(node, eh)
        self.handleBlock(then_list)
        else_list = node.orelse
        if len(else_list) == 0:
            self.add_edge(node, nnode)
        else:
            self.add_edge(node, else_list[0])
            self.add_edge(else_list[-1], nnode)
        self.handleBlock(else_list)
        self.generic_visit(node)

    def visit_Return(self, node):
        """add and edge to the last on return"""
        self.add_edge(node, self._last)

    def visit_Continue(self, node):
        """add an edges back to the loop"""
        self.add_edge(node, self._current_loop[-1])

    def visit_Break(self, node):
        """go to the next target"""
        self.add_edge(node, self._next_loop_target[-1])

    def visit_With(self, node):
        """in a with just treat it as another block"""
        nnode = self.get_target(node)
        self._init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], nnode)
        self.handleBlock(node.body)
        self.generic_visit(node)

    def buildCFG(self, visited, statement):
        """build the final CFG with boths succsessors and predicenents"""
        if statement in visited:
            return
        visited.add(statement)
        # get all in the map
        if statement in self._init_map:
            vals = self._init_map[statement]
        else:
            vals = []
        for succsessor in vals:
            if statement in self._succs:
                self._succs[statement].add(succsessor)
            else:
                self._succs[statement] = set()
                self._succs[statement].add(succsessor)

            if succsessor in self._preds:
                self._preds[succsessor].add(statement)
            else:
                self._preds[succsessor] = set()
                self._preds[succsessor].add(statement)
                self.buildCFG(visited, succsessor)

    def add_edge(self, from_node, to_node):
        """add an edge to the node in the initial map"""
        # don't do return unless it is going home
        if isinstance(from_node, ast.Return) and to_node != self._last:
            return
        # handle weird break case to make sure it doesn't go back to loop
        nlt = list(self._next_loop_target)
        # no next loop target. get off of the because you dont want to add yet
        if len(nlt) == 0 and isinstance(from_node, ast.Break):
            return
        # make sure we are sending a break to the correct node only
        if isinstance(from_node, ast.Break) and to_node != nlt[-1]:
            return
            # handle weird break case to make sure it doesn't go back to loop
        nlt = list(self._current_loop)
        # no next loop target. get off of the because you dont want to add yet
        if len(nlt) == 0 and isinstance(from_node, ast.Continue):
            return
        if isinstance(from_node, ast.Continue) and to_node != nlt[-1]:
            return
            # add the nodes to the statement list
        self._statements.add(from_node)
        self._statements.add(to_node)
        # add it to the initial map we have
        if from_node not in self._init_map:
            self._init_map[from_node] = set()
            self._init_map[from_node].add(to_node)
        else:
            self._init_map[from_node].add(to_node)

    def handleBlock(self, node_list):
        """handle a list of blocks.  We assume
        that the first and last nodes in this block
        are already pointed at correctly and have the correct
        targets"""
        if len(node_list) == 0:
            # if there is nothing than nothing to do here
            return
        elif len(node_list) == 1:
            # if it is only one node than add edges to the execution handlers
            for target in self._try_targets:
                self.add_edge(node_list[0], target)
        else:
            # otherwise we need to loop through them all and add edges
            from_node = node_list[0]
            for to_node in node_list[1:]:
                if isinstance(from_node, ast.Break):
                    pass
                elif isinstance(from_node, ast.Continue):
                    pass
                else:
                    self.add_edge(from_node, to_node)

                # handle exception handlers
                for target in self._try_targets:
                    self.add_edge(from_node, target)
                from_node = to_node

            # exception handing for the last node in the list
            for target in self._try_targets:
                self.add_edge(to_node, target)

    def handle_loop(self, node):
        """handle a loop construct"""
        # some housekeeping here to handel current loop and next loop for break
        # and continue statemnts
        # keep track of these for break and continue statements
        nnode = self.get_target(node)
        self._current_loop.append(node)
        self._next_loop_target.append(nnode)
        # handle or else clauses on loops lol python
        if len(node.orelse) > 0:
            self.add_edge(node, node.orelse[0])
            self.add_edge(node.orelse[-1], nnode)
        # handle excpetions
        for eh in self._try_targets:
            self.add_edge(node, eh)
        # point the loop to the right stuff
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node)
        # visit blocks and continue
        self.handleBlock(node.body)
        self.handleBlock(node.orelse)
        self.generic_visit(node)
        # get rid of some bookkeeping stuff
        self._current_loop = self._current_loop[:-1]
        self._next_loop_target = self._next_loop_target[:-1]

    def get_target(self, node):
        """get the target of a node from the init map
        only returns non exception handlers which
        may already be in the map.  Don't worry they will
        get added back to the map if needed"""
        nnode = list(self._init_map[node])
        if len(nnode) != 1:
            # find true target
            for i in nnode:
                if not isinstance(i, ast.ExceptHandler):
                    nnode = i
                    break
        else:
            nnode = nnode[0]
        return nnode


class BuildAllCFG(ast.NodeVisitor):

    """build cfg for all functions"""

    def __init__(self, verbose=True, code=None):
        self.verbose = verbose
        self.store = dict()
        self.current_key = None
        self.code = code

    def visit_Module(self, node):
        self.current_key = node
        self.store[node] = dict()
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        old_key = self.current_key
        self.current_key = node
        self.store[node] = dict()
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        func_visit = CFGVisitor(node)
        func_visit.start_visit(node)
        for key, value in func_visit.stores.iteritems():
            self.store[self.current_key][key] = value
            if self.verbose:
                print '\n'
                if self.code is not None:
                    print 'CFGs for: ', key.lineno, self.code[key.lineno - 1].lstrip().strip()
                    print 'Forward: '
                    self.print_graph_code(value.succs)
                    print '\nBackward:'
                    self.print_graph_code(value.preds)
                else:
                    print 'CFGs for:', key.lineno, key
                    print 'Forward: '
                    self.print_graph(value.succs)
                    print '\nBackward:'
                    self.print_graph(value.preds)
                    print '\n'

    def print_graph_code(self, thing):
        vals = []
        for k, v in thing.iteritems():
            vals.append((k, sorted(v, key=lambda x: x.lineno)))
            vals = sorted(vals, key=lambda x: x[0].lineno)
        for node, edges in vals:
            print node.lineno, self.code[node.lineno - 1].lstrip().strip(), node
            for target in edges:
                print '\t', target.lineno, self.code[target.lineno - 1].lstrip().strip()

    def print_graph(self, thing):
        vals = []
        for k, v in thing.iteritems():
            vals.append((k, sorted(v, key=lambda x: x.lineno)))
            vals = sorted(vals, key=lambda x: x[0].lineno)
        for node, edges in vals:
            print node.lineno, node
            for target in edges:
                print '\t', target.lineno, target


def build_files_cfgs(tree=None, fname=None, verbose=False, src_code=None):
    """build a files cfg.  Either analyze a passed tree
        or analyze a passed file name.  Verbose to print
        as you go"""
    if tree is not None:
        cfgvisit = BuildAllCFG(verbose, code=src_code)
        cfgvisit.visit(tree)
        return cfgvisit.store
    elif fname is not None:
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)
            cfgvisit = BuildAllCFG(verbose, code=code.split('\n'))
            cfgvisit.visit(tree)
            return cfgvisit.store
    else:
        print 'No tree or file name passed!'
        return None


def main(fname, verbose):
    """main function
    :type fname: String
    :type verbose: Boolean
    """

    if os.path.isfile(fname):
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)
            cfgvisit = BuildAllCFG(code=code.split('\n'))
            cfgvisit.visit(tree)

    else:
        print 'error no file'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('This is a program to find'
                                                  'constant thresholds in a python program'))
    parser.add_argument('file', help='path to file')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    main(args.file, args.verbose)
