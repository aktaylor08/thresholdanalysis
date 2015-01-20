from __future__ import print_function

import ast
from ast_tools import get_name
import sys


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
        func = cfg.start.function_node
        # func = list(cfg.preds[cfg.start])[0]
        # handle the arguments
        arguments = func.args.args
        for arg in arguments:
            if isinstance(arg, ast.Name):
                if arg.id == 'self':
                    pass
                else:
                    outs[cfg.start].add((arg.id, arg))
            else:
                print(func.lineno, arg)
                print('ERROR argument unsupported type', file=sys.stderr)

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
        gen = self.get_gen(node)
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

    @staticmethod
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
