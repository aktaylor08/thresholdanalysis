from collections import defaultdict, deque
import sys
from cfg_analysis import BuildAllCFG, FunctionCall, FunctionReturn
from ast_tools import get_name
import ast
from reaching_definition import ReachingDefinition

__author__ = 'ataylor'


class AnalysisGraph(object):
    """"Graph which contains links to all of the classes
    in a file that have been analyzed using the program
    basically a dictonary with some functionality that may be
    added to it depending on what is needed.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.classes = {"__GLOBAL__": ClassGraph(None)}

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


    def import_rd(self, reach_defs):
        """THIS MUST BE RAN AFTER CONTROL FLOW ANALYSIS"""
        for i in reach_defs.keys():
            if isinstance(i, ast.Module):
                class_graph = self.classes["__GLOBAL__"]
                class_graph.import_rd(reach_defs[i])
            else:
                class_graph = self.classes[i.name]
                class_graph.import_rd(reach_defs[i])




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
        self.rd = {}
        self.calls = []

    def is_global(self):
        """Return true if this is in the global namespace and not a class per se"""
        return self.node is None

    def import_rd(self, class_rd):
        function_defs = defaultdict(set)
        for function in class_rd:
            for statement in class_rd[function]:
                rd = self.rd.get(statement, dict())
                for stmt, deff in class_rd[function][statement]:
                    var_defs = rd.get(stmt, list())
                    var_defs.append(deff)
                    rd[stmt] = var_defs
                    if stmt.startswith('self.'):
                        function_defs[stmt].add(deff)
                self.rd[statement] = rd
        for i in function_defs.keys():
            print i
            for j in list(function_defs[i]):
                print '\t', j.lineno, j


        # now add class rd's so self.x are correctly linked everywhere that is not a direct sign
        for c in self.rd:
            for stmt, keys in self.rd.iteritems():
                print stmt,keys
                for name, org_things in keys.iteritems():
                    print name, org_things

        exclude = {}



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
            # check for all calls
            call = None
            if isinstance(pos_call, ast.Call):
                call = pos_call
            if isinstance(pos_call, ast.Expr):
                if isinstance(pos_call.value, ast.Call):
                    call = pos_call.value
            if call is not None:
                name = get_name(call.func)
                start = None
                end = None
                if self.is_global():
                    start = self.get_start(name)
                    end = self.get_last(name)
                elif name.startswith('self.'):
                    start = self.get_start(name[5:])
                    end = self.get_last(name[5:])

                # Not in the class or namespace so we are good and done
                if start is None:
                    continue

                # take care of the forward cfg first
                func_call_obj = FunctionCall(call, None)
                func_return_obj = FunctionReturn(call, None)
                if func_call_obj not in self.calls:
                    # clear the old targets and point at the call
                    old_targets = list(self.cfg_forward[pos_call])
                    self.cfg_forward[pos_call].clear()
                    self.cfg_forward[pos_call].add(func_call_obj)

                    # point it at the start of the function
                    self.cfg_forward[func_call_obj].add(start)

                    # take care of return values
                    self.cfg_forward[end].add(func_return_obj)
                    for target in old_targets:
                        self.cfg_forward[func_return_obj].add(target)
                    self.calls.append(func_call_obj)

                    # now do backwards stuff
                    for i in old_targets:
                        self.cfg_backward[i].remove(pos_call)
                        self.cfg_backward[i].add(func_return_obj)
                    self.cfg_backward[func_call_obj].add(pos_call)
                    self.cfg_backward[start].add(func_call_obj)
                    self.cfg_backward[func_return_obj].add(end)

    def print_cfg(self, func):
        to_visit = deque()
        visited = set()
        to_visit.append(self.get_start(func))
        print "Forward CFG for:", func
        while len(to_visit) > 0:
            next_node = to_visit.popleft()
            targets = self.cfg_forward[next_node]
            print '\t', next_node.lineno, next_node
            for i in targets:
                if i not in visited:
                    to_visit.append(i)
                print '\t\t', i.lineno, i
            visited.add(next_node)
        visited.clear()
        to_visit.append(self.get_last(func))

        print "\n\nBackward CFG for:", func
        while len(to_visit) > 0:
            next_node = to_visit.popleft()
            targets = self.cfg_backward[next_node]
            print '\t', next_node.lineno, next_node
            for i in targets:
                if i not in visited:
                    to_visit.append(i)
                print '\t\t', i.lineno, i
            visited.add(next_node)
        visited.clear()

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


def main(file_name):
    with open(file_name) as openf:
        code = openf.read()
        tree = ast.parse(code)
        cfgvisit = BuildAllCFG(False, code=code.split('\n'))
        cfgvisit.visit(tree)
        rd = ReachingDefinition(tree, cfgvisit.store)
        rd.compute()

        ag = AnalysisGraph(file_name)
        ag.import_cfg(cfgvisit.store)
        ag.import_rd(rd.rds_in)


if __name__ == "__main__":
    fname = sys.argv[1]
    main(fname)
