#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import pprinter

class CFGVisitor(ast.NodeVisitor):
    '''build the CFG for a function in python''' 

    def __init__(self, func ):
        '''initilize the function'''
        self.statements = set()
        self.init_map = dict()
        self.func =  func
        self.start = None
        self.last = func
        self.succs = dict()
        self.preds = dict()
        self.current_loop = [] 
        self.next_loop_target= [] 
        self.try_targets = []




    def visit_FunctionDef(self, node):
        ''' here lies the start of it all.  Add some bookkeeping edges'''
        #set the start to be the first node in body
        self.start = node.body[0]
        #the last one is the function itself
        #now start the cfg by adding one to the end
        self.last = node

        #one node body is also different
        self.add_edge(node.body[-1], self.last)
        self.handleBlock(node.body)
        self.generic_visit(node)
        self.buildCFG(set(), self.start)



    def visit_While(self, node):
        '''pass it to the loop handler'''
        self.handle_loop(node)


    def visit_For(self, node):
        '''pass it to the loop handler'''
        self.handle_loop(node)


    def visit_TryFinally(self, node):
        '''visit a try finally block'''
        #get where this node points
        nnode = self.get_target(node)
        #clear the nodes target and point it at the body
        self.init_map[node].clear()

        #point to the correct things
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node.finalbody[0])
        self.add_edge(node.finalbody[-1], nnode)
        
        #handle both blocks and continue on
        self.handleBlock(node.body)
        self.handleBlock(node.finalbody)
        self.generic_visit(node)


    def visit_TryExcept(self, node):
        '''try excpts'''
        target = self.get_target(node)
        #point to some of the right things
        self.init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], target)

        if len(node.handlers) == 1:
            #if only once except than point that one to the target
            self.add_edge(node.handlers[0], target)
        else:
            #otherwise add all of the nodes and  stuff
            first = node.handlers[0]
            self.add_edge(first,target)
            for nh in node.handlers[1:]:
                self.add_edge(first,nh)
                self.add_edge(nh, target)
                first = nh
        
        #add exception handlers to be pointed at only point to first one
        self.try_targets.append(node.handlers[0])
        
        #visit
        self.handleBlock(node.body)
        self.generic_visit(node)
        #remove them
        self.try_targets = self.try_targets[:-1]
        

    def visit_ExceptHandler(self, node):
        '''here we visit an exception handler'''
        target = self.get_target(node)

        #we are good so 
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], target)
        #pop one off before we visit some blocks because they will be
        #already visited and don't want to point to ourselves
        tt = self.try_targets[-1]
        self.try_targets = self.try_targets[:-1]
        self.handleBlock(node.body)
        self.try_targets.append(tt)
        self.generic_visit(node)




    def visit_If(self, node):
        '''visit an if node'''
        nnode = self.get_target(node)
        self.init_map[node].clear()
        #do stuff with the then list
        then_list = node.body
        self.add_edge(node, then_list[0])
        for eh in self.try_targets:
            self.add_edge(node,eh)
        self.add_edge(then_list[-1], nnode)
        #handle try targets
        for eh in self.try_targets:
            self.add_edge(node,eh)
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
        '''add and edge to the last on return'''
        self.add_edge(node, self.last)


    def visit_Continue(self, node):
        '''add an edges back to the loop'''
        self.add_edge(node, self.current_loop[-1])


    def visit_Break(self, node):
        '''go to the next target'''
        self.add_edge(node, self.next_loop_target[-1])


    def visit_With(self, node):
        '''in a with just treat it as another block'''
        nnode = self.get_target(node)
        self.init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], nnode)
        self.handleBlock(node.body)
        self.generic_visit(node)


    def buildCFG(self, visited, statement):
        '''build the final CFG with boths succsessors and predicenents'''
        if statement in visited:
            return
        visited.add(statement)
        #get all in the map
        if statement in self.init_map:
            vals = self.init_map[statement]
        else:
            vals = []
        for succsessor in vals:
            if statement in self.succs:
                self.succs[statement].add(succsessor)
            else:
                self.succs[statement] = set()
                self.succs[statement].add(succsessor)

            if succsessor in self.preds:
                self.preds[succsessor].add(statement)
            else:
                self.preds[succsessor] = set()
                self.preds[succsessor].add(statement)
                self.buildCFG(visited, succsessor)


        



        
    def add_edge(self,from_node, to_node):
        '''add an edge to the node in the initial map''' 
        #don't do return unless it is going home 
        if isinstance(from_node, ast.Return) and to_node != self.last:
            return
        #handle weird break case to make sure it doesn't go back to loop
        nlt = list(self.next_loop_target)
        #no next loop target. get off of the because you dont want to add yet
        if len(nlt) == 0 and isinstance(from_node, ast.Break):
            return
        #make sure we are sending a break to the correct node only
        if isinstance(from_node, ast.Break) and to_node != nlt[-1]:
            return 
        #handle weird break case to make sure it doesn't go back to loop
        nlt = list(self.current_loop)
        #no next loop target. get off of the because you dont want to add yet
        if len(nlt) == 0 and isinstance(from_node, ast.Continue):
            return
        if isinstance(from_node, ast.Continue) and to_node != nlt[-1]:
            return 
        #add the nodes to the statement list
        self.statements.add(from_node)
        self.statements.add(to_node)
        #add it to the initial map we have
        if from_node not in self.init_map:
            self.init_map[from_node] = set() 
            self.init_map[from_node].add(to_node)
        else:
            self.init_map[from_node].add(to_node)


    def handleBlock(self, node_list):
        '''handle a list of blocks.  We assume 
        that the first and last nodes in this block
        are already pointed at correctly and have the correct
        targets'''
        if len(node_list) == 0:
            #if there is nothing than nothing to do here
            return
        elif len(node_list) == 1:
            # if it is only one node than add edges to the execption handlers
            for target in self.try_targets:
                self.add_edge(node_list[0], target)
        else:
            #otherwise we need to loop through them all and add edges
            from_node = node_list[0]
            for to_node in node_list[1:]:
                if isinstance(from_node, ast.Break):
                    pass
                elif isinstance(from_node, ast.Continue):
                    pass
                else:
                    self.add_edge(from_node, to_node)

                #handle exception handlers
                for target in self.try_targets:
                    self.add_edge(from_node, target)
                from_node = to_node

            #exception handing for the last node in the list
            for target in self.try_targets:
                self.add_edge(to_node, target)


    def handle_loop(self, node):
        '''handle a loop construct'''
        #some housekeeping here to handel current loop and next loop for break
        #and continue statemnts
        #keep track of these for break and continue statements
        nnode = self.get_target(node)
        self.current_loop.append(node)
        self.next_loop_target.append(nnode)
        #handle or else clauses on loops lol python
        if len(node.orelse) > 0:
            self.add_edge(node, node.orelse[0])
            self.add_edge(node.orelse[-1], nnode)
        #handle excpetions
        for eh in self.try_targets:
            self.add_edge(node,eh)
        #point the loop to the right stuff
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node)
        #visit blocks and continue
        self.handleBlock(node.body)
        self.handleBlock(node.orelse)
        self.generic_visit(node)
        #get rid of some bookkeeping stuff
        self.current_loop = self.current_loop[:-1]
        self.next_loop_target = self.next_loop_target[:-1]


    def get_target(self, node):
        '''get the target of a node from the init map
        only returns non exception handlers which
        may already be in the map.  Don't worry they will
        get added back to the map if needed'''
        nnode = list(self.init_map[node])
        if len(nnode) != 1:
            #find true target
            for i in nnode:
                if not isinstance(i, ast.ExceptHandler):
                    nnode = i
                    break
        else:
            nnode = nnode[0] 
        return nnode





class BuildAllCFG(ast.NodeVisitor):
    '''build cfg for all functions'''

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.store = dict()
        self.current_key = None 

    def visit_Module(self, node):
        print 'hi'
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
        func_visit.visit(node)
        self.store[self.current_key][node] = func_visit
        if self.verbose:
            print '\n'
            print '\n'
            print '!*!*!*!*!' * 5 
            print '\n'
            print '-----Forward CFG for----- ', node.lineno, node
            print '\n'
            print_graph(func_visit.succs)
            print '\n'
            print 'Backward CFG'
            print '\n'
            print_graph(func_visit.preds)


def print_graph(thing):
    vals = []
    for k,v in thing.iteritems(): 
        vals.append((k, sorted(v, key=lambda x: x.lineno)))
        vals = sorted(vals, key=lambda x: x[0].lineno)
    for node, edges in vals:
        print node.lineno, node
        for target in edges:
            print '\t', target.lineno, target
        

def build_files_cfgs(tree=None, fname=None, verbose=False):
    '''build a files cfg.  Either analyze a passed tree
        or analyze a passed file name.  Verbose to print
        as you go'''
    if tree is not None:
        cfgvisit = BuildAllCFG(verbose)
        cfgvisit.visit(tree)
        return cfgvisit.store
    elif fname is not None:
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  
            cfgvisit = BuildAllCFG(verbose) 
            cfgvisit.visit(tree)
            return  cfgvisit.store
    else:
        print 'No tree or file name passed!'
        return None

        






def main(fname):
    '''main function'''

    if os.path.isfile(fname):
        tree = None
        with open(fname, 'r') as openf:
            code = openf.read()
            tree = ast.parse(code)  
            cfgvisit = BuildAllCFG() 
            cfgvisit.visit(tree)


    else:
        print 'error no file'



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('This is a program to find' 
        'constant thresholds in a python program'))
    parser.add_argument('file',  help='path to file')
    args = parser.parse_args()
    main(args.file)
