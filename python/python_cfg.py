#!/usr/bin/env python
# encoding: utf-8

import ast
import os
import symtable
import argparse

import pprinter

class CFGVisitor(ast.NodeVisitor):
    '''build the CFG for a function'''


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

    def add_edge(self,from_node, to_node):
        '''add an edge to the node'''
        if isinstance(from_node, ast.Return) and to_node != self.last:
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
        if len(node_list) <= 1:
            return
        else:
            from_node = node_list[0]
            for to_node in node_list[1:]:
                if not isinstance(from_node, ast.Break) and not isinstance(from_node, ast.Continue):
                    self.add_edge(from_node, to_node)
                from_node = to_node


    def visit_FunctionDef(self, node):
        #set the start to be the first node in body
        self.start = node.body[0]
        #the last one is the function itself
        #now start the cfg by adding one to the end
        self.last = node

        #one node body is also different
        self.add_edge(node.body[-1], self.last)
        self.handleBlock(node.body)
        self.generic_visit(node)


    def visit_While(self, node):
        self.handle_loop(node)

    def visit_For(self, node):
        self.handle_loop(node)

    def handle_loop(self, node):
        #some housekeeping here to handel current loop and next loop for break
        #and continue statemnts
        self.current_loop.append(node)
        nnode = list(self.init_map.get(node))
        if len(nnode) != 1:
            print 'ERROR IN CFG LOOP CONSTURCTION\n\n\n\n\n\n' * 200
        self.next_loop_target.append(nnode[0])
        if len(node.orelse) > 0:
            self.add_edge(node, node.orelse[0])
            self.add_edge(node.orelse[-1], nnode[0])

        #point the loop to the right stuff
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node)

        self.handleBlock(node.body)
        self.handleBlock(node.orelse)
        self.generic_visit(node)

        self.current_loop = self.current_loop[:-1]
        self.next_loop_target = self.next_loop_target[:-1]


    def visit_If(self, node):
        next_node = list(self.init_map[node])
        if len(next_node) != 1:
            print 'ERRRORROROROROR'
        self.init_map[node].clear()
        next_node = next_node[0]

        #do stuff with the then list
        then_list = node.body
        self.add_edge(node, then_list[0])
        self.add_edge(then_list[-1], next_node)
        self.handleBlock(then_list)

        else_list = node.orelse
        if len(else_list) == 0:
            self.add_edge(node, next_node)
        else:
            self.add_edge(node, else_list[0])
            self.add_edge(else_list[-1], next_node)
        self.handleBlock(else_list)
        self.generic_visit(node)


    def visit_Return(self, node):
        self.add_edge(node, self.last)


    def visit_Continue(self, node):
        self.add_edge(node, self.current_loop[-1])


    def visit_Break(self, node):
        self.add_edge(node, self.next_loop_target[-1])



class BuildAllCFG(ast.NodeVisitor):

    def __init__(self):
        self.store = dict()
        self.current_key = 'GLOBAL'
        self.store[self.current_key] = dict()

    def visit_ClassDef(self, node):
        self.current_key = node.name
        self.store[self.current_key] = dict()
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        #print pprinter.dump(node)
        func_visit = CFGVisitor(node)
        func_visit.visit(node)
        self.store[self.current_key][node.name] = func_visit
        print '------'
        for k,v in func_visit.init_map.iteritems():
            print k.lineno, k
            for i in v:
                print '\t',i.lineno, i
        


        





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
