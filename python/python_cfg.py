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
        self.try_targets = []


    def add_edge(self,from_node, to_node):
        '''add an edge to the node'''
        if isinstance(from_node, ast.Return) and to_node != self.last:
            return
        
        #handle weird break case to make sure it doesn't go back to loop
        nlt = list(self.next_loop_target)
        #no next loop target. get off of the because you dont want to add yet
        if len(nlt) == 0 and isinstance(from_node, ast.Break):
            return

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
            return
        elif len(node_list) == 1:
            #stupid excpetions
            for target in self.try_targets:
                self.add_edge(node_list[0], target)
        else:
            from_node = node_list[0]
            for to_node in node_list[1:]:
                if isinstance(from_node, ast.Break):
                    pass
                elif isinstance(from_node, ast.Continue):
                    pass
                else:
                    self.add_edge(from_node, to_node)

                #stupid excpetions
                for target in self.try_targets:
                    self.add_edge(from_node, target)
                from_node = to_node
            #exception handing
            for target in self.try_targets:
                self.add_edge(to_node, target)


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


    def visit_TryFinally(self, node):
        nnode = self.get_target(node)


        self.init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node.finalbody[0])
        self.add_edge(node.finalbody[-1], nnode)
        self.handleBlock(node.body)
        self.handleBlock(node.finalbody)
        self.generic_visit(node)


    def visit_TryExcept(self, node):
        target = self.get_target(node)

        self.init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], target)

        if len(node.handlers) == 1:
            self.add_edge(node.handlers[0], target)
        else:
            #otherwise add all of the nodes and  stuff
            first = node.handlers[0]
            self.add_edge(first,target)
            for nh in node.handlers[1:]:
                self.add_edge(first,nh)
                self.add_edge(nh, target)
                first = nh
        
        #add them
        self.try_targets.append(node.handlers[0])
        self.handleBlock(node.body)
        self.generic_visit(node)
        #remove them
        self.try_targets = self.try_targets[:-1]
        

    def visit_ExceptHandler(self, node):
        target = self.get_target(node)
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], target)
        #pop one off before we visit some blocks
        tt = self.try_targets[-1]
        self.try_targets = self.try_targets[:-1]
        self.handleBlock(node.body)
        self.try_targets.append(tt)
        self.generic_visit(node)



    def handle_loop(self, node):
        #some housekeeping here to handel current loop and next loop for break
        #and continue statemnts
        self.current_loop.append(node)
        nnode = self.get_target(node)
        self.next_loop_target.append(nnode)

        if len(node.orelse) > 0:
            self.add_edge(node, node.orelse[0])
            self.add_edge(node.orelse[-1], nnode)

        #handle excpetions
        for eh in self.try_targets:
            self.add_edge(node,eh)
        #point the loop to the right stuff
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], node)

        self.handleBlock(node.body)
        self.handleBlock(node.orelse)
        self.generic_visit(node)

        self.current_loop = self.current_loop[:-1]
        self.next_loop_target = self.next_loop_target[:-1]


    def visit_If(self, node):
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
        self.add_edge(node, self.last)


    def visit_Continue(self, node):
        self.add_edge(node, self.current_loop[-1])


    def visit_Break(self, node):
        self.add_edge(node, self.next_loop_target[-1])

    def visit_With(self, node):
        print pprinter.dump(node)
        nnode = self.get_target(node)
        self.init_map[node].clear()
        self.add_edge(node, node.body[0])
        self.add_edge(node.body[-1], nnode)
        self.handleBlock(node.body)
        self.generic_visit(node)



        
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

    def __init__(self):
        self.store = dict()
        self.current_key = 'GLOBAL'
        self.store[self.current_key] = dict()

    def visit_ClassDef(self, node):
        self.current_key = node.name
        self.store[self.current_key] = dict()
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        func_visit = CFGVisitor(node)
        func_visit.visit(node)
        self.store[self.current_key][node.name] = func_visit
        print '------'
        vals = []
        for k,v in func_visit.init_map.iteritems():
            vals.append((k, sorted(v, key=lambda x: x.lineno)))
        vals = sorted(vals, key=lambda x: x[0].lineno)
        for node, edges in vals:
            print node.lineno, node
            for target in edges:
                print '\t', target.lineno, target
        


        





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
