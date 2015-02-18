#!/usr/bin/env python
import argparse

from collections import defaultdict, deque
import os
import sys
from cfg_analysis import BuildAllCFG, FunctionCall, FunctionReturn, FunctionEntrance, FunctionExit
from ast_tools import get_name, get_node_variables, ContainingVisitor, get_repr
import ast
from reaching_definition import ReachingDefinition
from backward_analysis import get_constants, get_const_control
from backward_analysis import get_pub_srv_calls
from instrumentation import instrument_thresholds, NodeFinder, get_names
import json


import networkx as nx
import matplotlib.pyplot as plt


class AnalysisGraph(object):
    """"Graph which contains links to all of the classes
    in a file that have been analyzed using the program
    basically a dictionary with some functionality that may be
    added to it depending on what is needed.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.classes = {"__GLOBAL__": ClassGraph(None)}

    def import_cfg(self, cfg_store):
        for cls in cfg_store:
            if isinstance(cls, ast.Module):
                class_graph = self.classes["__GLOBAL__"]
                class_graph.cls_node = cls
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

    def get_class(self, cls):
        if isinstance(cls, ast.ClassDef):
            return self.classes[cls.name]
        elif isinstance(cls, ast.Module):
            return self.classes['__GLOBAL__']
        elif isinstance(cls, str):
            return self.classes[cls]
        else:
            print "Error on class lookup"
            return None

    def add_constant_ctrl(self, key, values):
        for cls, graph in self.classes.iteritems():
            graph.add_constant(key, values)

    def add_pub_srv(self, call):
        for cls, graph in self.classes.iteritems():
            graph.add_pub_srv(call)

    def do_analysis(self, call):
        cls_graph = None
        for i in self.classes.values():
            if call.node in i.nodes:
                cls_graph = i
                call = call.node
                break
            elif call.expr in i.nodes:
                cls_graph = i
                call = call.expr
                break
        cls_graph.do_analysis(call)


class ClassGraph(object):
    """Class Graph.  This contains the cfg, and rd for the classes and
    functions within a class.  Makes searching these two objects easy
    by providing convienent methods.
    """

    def __init__(self, node):
        self.cls_node = node
        self.functions = set()
        self.nodes = set()
        self.cfg_forward = defaultdict(set)
        self.cfg_backward = defaultdict(set)
        self.first_last_store = {}
        self.cfg_store = {}
        self.rd = {}
        self.calls = []
        self.const_flow = defaultdict(list)
        self.pub_srvs = []
        self.thresholds = set()
        self.ba_paths = {}
        self.fa_paths = {}

    def is_global(self):
        """Return true if this is in the global namespace and not a class per se"""
        return self.cls_node is None

    def calc_distances(self, pub):
        queue = deque()
        visited = set()
        found_shortest = set()
        distances = {}
        queue.append((pub, 0))
        links = self.ba_paths[pub]
        graph = defaultdict(set)
        for i in links:
            graph[i[0]].add(i[1])
        while len(queue) > 0:
            current, distance = queue.popleft()
            if current in self.thresholds:
                if current not in found_shortest:
                    distances[current] = distance
                    found_shortest.add(current)
            for target in graph[current]:
                if target not in visited:
                    visited.add(target)
                    queue.append((target, distance + 1))
        return distances

    def import_rd(self, class_rd):
        class_reach_defs = defaultdict(set)
        for function in class_rd:
            for statement in class_rd[function]:
                rd = self.rd.get(statement, dict())
                for stmt, deff in class_rd[function][statement]:
                    var_defs = rd.get(stmt, list())
                    var_defs.append(deff)
                    rd[stmt] = var_defs
                    if stmt.startswith('self.'):
                        class_reach_defs[stmt].add(deff)
                self.rd[statement] = rd

        # now add class rd's so self.x are correctly linked everywhere that is not a direct sign
        for stmt, reaching_defs in self.rd.iteritems():
            for class_var, class_rd in class_reach_defs.iteritems():
                if class_var in reaching_defs:
                    for i in class_rd:
                        reaching_defs[class_var].append(i)
                else:
                    nv = class_rd
                    reaching_defs[class_var] = set(nv)

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

    def do_analysis(self, node):
        self.ba_paths[node] = set()

        to_visit = deque()
        to_visit.append(node)
        visited = set()

        while len(to_visit) > 0:
            next_node = to_visit.popleft()
            new_nodes = []
            # get data dependencies
            used_variables = get_node_variables(next_node)
            for used_var in used_variables:

                # get reaching definitions
                rd = self.rd[next_node]
                keys = [x for x in rd]
                locations = set()
                for pos in keys:
                    if pos == used_var:
                        for x in rd[pos]:
                            locations.add(x)
                    if pos.startswith(used_var):
                        for x in rd[pos]:
                            locations.add(x)
                for assignment in locations:
                    if isinstance(assignment, FunctionEntrance):
                        self.handle_function_data_flow(assignment, used_var)
                    else:
                        new_nodes.append(assignment)

            # get data flow dependencies
            cf_nodes = self.get_flow(next_node)
            for i in cf_nodes:
                new_nodes.append(i)

            # deal with new nodes
            for i in new_nodes:
                self.ba_paths[node].add((next_node, i))
                if i not in visited and i not in to_visit:
                    to_visit.append(i)

            if next_node in self.const_flow:
                self.thresholds.add(next_node)
            visited.add(next_node)

    def get_flow(self, node):
        cv = ContainingVisitor(node)
        cv.visit(self.cls_node)
        if isinstance(cv.res, ast.FunctionDef):
            call = self.get_start(cv.res)
            ret_val = []
            for i in self.cfg_backward[call]:
                self.cfg_backward[call]
                for j in self.cfg_backward[i]:
                    ret_val.append(j)
            return ret_val
        else:
            return [cv.res]

    def handle_function_data_flow(self, assignment, used_var):
        # TODO Handle data across functions?
        # print 'Functional data flow', assignment, used_var
        pass

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

    def add_constant(self, key, values):
        for node in self.nodes:
            if node == key:
                for val in values:
                    if isinstance(val, ast.Num):
                        const = val
                    else:
                        const = val
                    if const not in self.const_flow[node]:
                        self.const_flow[node].append(const)

    def add_pub_srv(self, call):
        for node in self.nodes:
            if node == call.node or node == call.expr:
                if node not in self.pub_srvs:
                    self.pub_srvs.append(node)

    def draw_graph(self, forward=True):
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        if forward:
            cfg = self.cfg_forward
        else:
            cfg = self.cfg_backward
        for i,j in cfg.iteritems():
            for k in list(j):
                G.add_edge(i,k)
        pos = nx.graphviz_layout(G, prog='dot')
        nx.draw_networkx_edges(G,pos)
        nx.draw_networkx_nodes(G,pos)
        labs = nx.draw_networkx_labels(G,pos)

        for k,v in labs.iteritems():
            if isinstance(k, FunctionCall):
                v.set_text('call: ' + str(k.lineno))
            elif isinstance(k, FunctionReturn):
                v.set_text('return: ' + str(k.lineno))
            elif isinstance(k, FunctionEntrance):
                v.set_text('entrance: ' + str(k.lineno))
            elif isinstance(k, FunctionExit):
                v.set_text('exit: ' + str(k.lineno))
            else:
                v.set_text(k.lineno)

        plt.show()

    def graph_ba(self):
        thresh_lines = [x.lineno for x in self.thresholds]
        for i,k in self.ba_paths.iteritems():
            G = nx.DiGraph()
            temp_thresh = set()
            others = set()
            for link in list(k):
                if link[0].lineno in thresh_lines:
                    temp_thresh.add(link[0].lineno)
                else:
                    others.add(link[0].lineno)
                if link[1].lineno in thresh_lines:
                    temp_thresh.add(link[1].lineno)
                else:
                    others.add(link[1].lineno)
                G.add_node(link[0].lineno)
                G.add_node(link[1].lineno)
                G.add_edge(link[0].lineno, link[1].lineno)
            pos=nx.graphviz_layout(G, prog='dot')
            nx.draw_networkx_edges(G, pos)
            nx.draw_networkx_nodes(G, pos, nodelist=list(temp_thresh), node_color='b')
            nx.draw_networkx_nodes(G, pos, nodelist=list(others), node_color='r')
            nx.draw_networkx_labels(G, pos)
            plt.show()

class TestInfoVisitor(ast.NodeVisitor):
    """Class to transform the node into something
        that instead of doing calculations pulls them out into all
        other

    """

    def __init__(self, thresholds):
        self.things = []
        self.thresholds = thresholds

        # keep track of things
        self.information = {'comp': [], 'thresh': [], 'res': [], 'vals': [], 'boolop': []}
        self.lambda_dic = {}

        self.op_map = {}
        self.comparisions = []

        self.comps = deque()

        # keep track of numberings
        self.cnum = 0
        self.tnum = 0
        self.rnum = 0
        self.vnum = 0
        self.checknum = 0

    def create_boolop(self, node):
        n = 'check_{:d}'.format(self.checknum)
        self.checknum += 1
        self.information['boolop'].append(n)
        return n

    def create_thresh(self, node):
        n = 'thresh_{:d}'.format(self.tnum)
        self.tnum += 1
        self.information['thresh'].append(n)
        return n

    def create_comp(self, node):
        n = 'cmp_{:d}'.format(self.tnum)
        self.cnum += 1
        self.things.append(n)
        self.information['comp'].append(n)
        return n

    def create_val(self, node):
        n = 'value_{:d}'.format(self.vnum)
        self.vnum += 1
        self.things.append(n)
        self.information['vals'].append(n)
        return n

    def create_res(self, node):
        n = 'res_{:d}'.format(self.rnum)
        self.rnum += 1
        self.things.append(n)
        self.information['res'].append(n)
        return n

    def visit_BoolOp(self, node):
        bo = self.create_boolop(node)
        slength = len(self.comps)
        for idx, val in enumerate(node.values):
            # On boolean op if one of the values contains a threshold than we need to visit and transform that node
            if self.check_contains(val):
                self.visit(val)
            # otherwise we can encapsulate the whole value in a name
            else:
                v = self.create_val(val)
        diff = len(self.comps) - slength
        if diff == 1:
            self.op_map[bo] = {'size': 1, 'op': '', 'comps': [self.comps.popleft()]}
        else:
            d = {'size': diff, 'comps': []}
            for _ in range(diff):
                d['comps'].append(self.comps.popleft())
            if isinstance(node.op, ast.And):
                d['op'] = 'and'
            elif isinstance(node.op, ast.Or):
                d['op'] = 'or'
            elif isinstance(node.op, ast.Or):
                d['op'] = 'unknown'
            self.op_map[bo] = d

    def visit_Compare(self, node):
        # Test to see if there is any part of the node that contains the thershold.  If not just replace the node
        cdict = {'thresh': [], 'cmp': [], 'res': ''}
        if not self.check_contains(node):
            print 'Its not in there?'
            return

            v = self.create_val(node)
        # now we need to loop through and replace stuff
        if node.left in self.thresholds:
            t = self.create_thresh(node.left)
            cdict['thresh'].append(t)
        elif self.check_contains(node.left):
            self.visit(node.left)
        else:
            c = self.create_comp(node.left)
            cdict['cmp'].append(c)
        for idx, val in enumerate(node.comparators):
            if val in self.thresholds:
                t = self.create_thresh(val)
                cdict['thresh'].append(t)
            elif self.check_contains(val):
                self.visit(val)
            else:
                c = self.create_comp(node.left)
                cdict['cmp'].append(c)
        r = self.create_res(node)
        cdict['res'] = r
        self.comps.appendleft(cdict)
        self.comparisions.append(cdict)

    def visit_UnaryOp(self, node):
        if node.operand in self.thresholds:
            cdict = {'thresh': [], 'cmp': [], 'res': ''}
            t = self.create_thresh(node.operand)
            c = self.create_comp(node.operand)
            res = self.create_res(node.operand)
            cdict['res'] = res
            cdict['thresh'].append(t)
            cdict['cmp'].append(c)
            self.comps.appendleft(cdict)
            self.comparisions.append(cdict)
        elif self.check_contains(node.operand):
            self.visit(node.operand)
        return node

    def check_contains(self, node):
        v = NodeFinder(self.thresholds)
        v.visit(node)
        return v.found


def main(file_name):
    parser = argparse.ArgumentParser(description=("This is a program to find"
                                                  " constant thresholds in a python program"))
    parser.add_argument('file', help='path to file')
    parser.add_argument('-n', '--no_execute', help='Set execution to false',
                        action='store_true', )
    parser.add_argument('-g', '--graph', help='Graph the thresholds and stuff',
                        action='store_true', )
    parser.add_argument('rest', nargs='*')
    args = parser.parse_args()
    with open(args.file) as openf:
        code = openf.read()
        split_code = code.split('\n')
        tree = ast.parse(code)
        cfgvisit = BuildAllCFG(False, code=code.split('\n'))
        cfgvisit.visit(tree)
        rd = ReachingDefinition(tree, cfgvisit.store)
        rd.compute()

        ag = AnalysisGraph(file_name)
        ag.import_cfg(cfgvisit.store)
        ag.import_rd(rd.rds_in)

        constants = get_constants(tree, code, args.file, False)
        const_control, const_srces = get_const_control(constants, tree, code, args.file)
        for key, values in const_control.iteritems():
            ag.add_constant_ctrl(key, values)

        pubs = get_pub_srv_calls(tree, code)
        for i in pubs:
            ag.add_pub_srv(i)

        for i in pubs:
            ag.do_analysis(i)

        if args.graph:
            for i, k in ag.classes.iteritems():
                k.graph_ba()
        thresholds = dict()
        distances = dict()

        for i, k in ag.classes.iteritems():
            for thresh in k.thresholds:
                thresholds[thresh] = k.const_flow[thresh]
            for ps in k.pub_srvs:
                d_info = k.calc_distances(ps)
                for i in d_info:
                    if i in distances:
                        distances[i].append((ps, d_info))
                    else:
                        distances[i] = [(ps, d_info[i])]

        static_information = {}
        for thresh in thresholds.iterkeys():
            names = []
            sources = []
            for x in thresholds[thresh]:
                names.append(get_repr(x))
                val = const_srces[thresh][x]
                if len(val) > 1:
                    print 'ERROR SOURCE???'
                    print val
                sources.append(val[0])
            info = {'lineno': thresh.lineno, 'file': args.file}
            info['key'] = str(args.file) + ':' + str(info['lineno'])
            idx = thresh.lineno - 1
            line_code = split_code[idx].strip().lstrip()
            while not line_code.endswith(':'):
                idx += 1
                line_code += split_code[idx].strip().lstrip()
            info['source_code'] = line_code
            info['topic'] = 'unknown'
            info['distance'] = min([d[1] for d in distances[thresh]])
            info['sources'] = sources
            info['names'] = names

            tiv = TestInfoVisitor(thresholds[thresh])
            tiv.visit(thresh.test)
            info['num_comparisons'] = len(tiv.comparisions)
            info['opmap'] = tiv.op_map
            info['comparisons'] = tiv.comparisions
            info['thresh'] = tiv.information['thresh']
            info['res'] = tiv.information['res']
            info['comp'] = tiv.information['comp']
            static_information[info['key']] = info
        fname,_ = os.path.splitext(args.file)
        f = fname + '_thresh_info.json'
        with open(f, 'w') as json_out:
            json.dump(static_information, json_out, indent=1)

        if not args.no_execute:
            instrument_thresholds(tree, thresholds, args.file, code.split('\n'), False, args.rest)



if __name__ == "__main__":
    fname = sys.argv[1]
    main(fname)
