from collections import defaultdict
import sys
from cfg_analysis import BuildAllCFG
from ast_tools import get_name
import ast

__author__ = 'ataylor'


class AnalysisGraph(object):
    """"Graph which contains links to all of the classes
    in a file that have been analyzed using the program
    basically a dictonary with some functionality that may be
    added to it depending on what is needed.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.classes = {"__GLOBAL__" : ClassGraph(None)}

    def import_cfg(self, cfg_store):
        for cls in cfg_store:
            if isinstance(cls, ast.Module):
                class_graph = self.classes["__GLOBAL__"]
                class_graph.import_cfg(cfg_store[cls])
            elif cls.name in self.classes:
                class_graph = self.classes[cls.name]
                class_graph.import_cfg(cfg_store[cls])
            else:
                class_graph = ClassGraph(cls)
                self.classes[cls.name] = class_graph
                class_graph.import_cfg(cfg_store[cls])


class ClassGraph(object):
    """Class Graph.  This contains the cfg, and rd for the classes and
    functions within a class.  Makes searching these two objects easy
    by providing convienent methods.
    """

    def __init__(self, node):
        self.node = node
        self.functions = set()
        self.nodes = set()
        self.cfg_forward = defaultdict(set)
        self.cfg_backward = defaultdict(set)
        self.first_last_store = {}
        self.cfg_store = {}
        self.rd = None

    def is_global(self):
        """Return true if this is in the global namespace and not a class per se"""
        return self.node is None

    def import_cfg(self, cfg):
        """Given a cfg store import it
        into the graph so it can be used"""
        for func in cfg:
            self.functions.add(func)

            # Add all of the nodes and all of the connections in the original cfg
            self.cfg_store[func] = cfg[func]
            store = cfg[func]
            self.first_last_store[func] = (store.start, store.last)
            self.nodes.add(store.start)
            self.nodes.add(store.last)
            for node in store.succs:
                self.nodes.add(node)
                for target in store.succs[node]:
                    self.cfg_forward[node].add(target)

            for node in store.preds:
                self.nodes.add(node)
                for target in store.preds[node]:
                    self.cfg_backward[node].add(target)

        # Make links between functions to make the call graph nice and crazy :)
        for pos_call in list(self.nodes):
            #check for all calls
            call = None
            if isinstance(pos_call, ast.Call):
                call = pos_call.func
            if isinstance(pos_call, ast.Expr):
                if isinstance(pos_call.value, ast.Call):
                    call = pos_call.value.func
            if call is not None:
                name = get_name(call)
                start = None
                end = None
                if self.is_global():
                    pass
                elif name.startswith('self.'):
                    start = self.get_start(name[5:])
                    end = self.get_last(name[5:])

                # Not in the class or namespace so we are good and done
                if start is None:
                    continue
                print self.cfg_forward[pos_call]
                old_targets = self.cfg_forward[pos_call]






    def get_start(self, function):
        """Get the start of the functions CFG"""
        if isinstance(function, str):
            for i in list(self.functions):
                if i.name == function:
                    function = i
                    break
        try:
            return self.first_last_store[function][0]
        except KeyError:
            return None

    def get_last(self, function):
        """Get the last of the functions CFG"""
        if isinstance(function, str):
            for i in self.functions:
                if i.name == function:
                    function = i
        try:
            return self.first_last_store[function][1]
        except KeyError:
            return None





def main(fname):
    with open(fname) as openf:
        code = openf.read()
        tree = ast.parse(code)
        cfgvisit = BuildAllCFG(False, code=code.split('\n'))
        cfgvisit.visit(tree)

        ag = AnalysisGraph(fname)
        ag.import_cfg(cfgvisit.store)
        print ag.classes['Gripper'].get_start('set_parameters')


if __name__ == "__main__":
    fname = sys.argv[1]
    main(fname)

