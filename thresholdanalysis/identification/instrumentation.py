import ast
import sys

import pprinter



# In this module I have botht the older version of instrumentation and the new version of isntrumentation.
# The old version is the replace values series of calls and is at the bottom of the file.as
# The new version is at the top and only exports select information that just reports threshold values
# and comparision values in the code

def instrument_thresholds(tree, thresholds, keys, fname, code, verbose, args):
    tree = InstrumentVisitor(thresholds, keys, fname, code, verbose).visit(tree)
    tree = add_import_statement(tree)
    tree = ast.fix_missing_locations(tree)
    code = compile(tree, fname, mode='exec')
    sys.argv = [fname] + args
    ns = {'__name__': '__main__'}
    exec (code, ns)


def add_import_statement(node):
    """Add the import statement to the top of the code before anything is done"""
    new_node = ast.Import(names=[ast.alias(name='reporting', asname=None)])
    new_node = ast.copy_location(new_node, node.body[0])

    ast.increment_lineno(node.body[0], 1)
    node.body = [new_node] + node.body
    return node


class InstrumentVisitor(ast.NodeTransformer):
    """Node transformer to instrument the code"""

    def __init__(self, thresholds, keys, fname, code, verbose):
        """setup"""
        self.fname = fname
        self.tmap = {}
        self.code = code
        self.keys = keys
        self.verbose = verbose
        self.debug = False
        for i in thresholds:
            self.tmap[i] = thresholds[i]

    def visit_If(self, node):
        """Just call the handle runtime function"""
        self.generic_visit(node)
        node = self.handle_node(node)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node = self.handle_node(node)
        return node

    def handle_node(self, node):
        if node in self.tmap:
            if isinstance(node.test, ast.Compare):
                pass
            elif isinstance(node.test, ast.BoolOp):
                pass
            elif isinstance(node.test, ast.UnaryOp):
                pass
            else:
                print type(node.test)
                print ast.dump(node.test)
                print 'Unexpected type here'
                return node

            ccollector = ComparisionCollector(self.tmap[node], node.lineno, self.keys[node])

            val = ccollector.visit(node.test)

            # have created the two required things here.

            args = []
            names = []
            for i in get_names(val):
                args.append(ast.Name(id=i.id, ctx=ast.Param(), lineno=node.lineno))
                names.append(i.id)
            args = ast.arguments(args=args, vararg=None, kwarg=None, defaults=[])
            result_lambda = ast.Lambda(args=args, body=val, lineno=node.lineno)
            ccollector.lambda_dic['result'] = names

            if self.debug:
                print pprinter.dump(val)
                for i in ccollector.information['vals']:
                    print '\t', ast.dump(i)
                for i in ccollector.information['comp']:
                    print '\t', ast.dump(i)
                for i in ccollector.information['thresh']:
                    print '\t', ast.dump(i)
                for i in ccollector.information['res']:
                    print '\t', ast.dump(i)
                print '\n\n\n\n'

            keys = []
            values = []
            for i in ccollector.lambda_dic:
                keys.append(ast.Str(s=i))
                elements = []
                for val in ccollector.lambda_dic[i]:
                    elements.append(ast.Str(s=val))
                values.append(ast.List(elts=elements, ctx=ast.Load()))
            arg_dict = ast.Dict(keys=keys, values=values)
            # add the function call
            name = ast.Name(id='reporting', ctx=ast.Load(lineno=node.lineno), lineno=node.lineno)
            attr = ast.Attribute(value=name, attr='report', ctx=ast.Load(lineno=node.lineno), lineno=node.lineno)

            report_vals = []
            key_list = []
            for i in ccollector.value_map:
                s = ast.Str(s=ccollector.value_map[i].key, lineno=node.lineno, col_offset=0)
                key_list.append(s)
                keys = [
                    ast.Str(s='cmp', lineno=node.lineno, col_offset=0),
                    ast.Str(s='thresh', lineno=node.lineno, col_offset=0),
                    ast.Str(s='res', lineno=node.lineno, col_offset=0),
                ]
                values = [
                    ast.Str(ccollector.value_map[i].comp.arg),
                    ast.Str(ccollector.value_map[i].thresh.arg),
                    ast.Str(ccollector.value_map[i].res.arg),
                ]
                one_dict = ast.Dict(keys=keys, values=values, lineno=node.lineno, col_offset=0)
                report_vals.append(one_dict)

            keys_arg = ast.List(elts=key_list, ctx=ast.Load(), lineno=node.lineno, col_offset=0)
            keys_arg = ast.fix_missing_locations(keys_arg)
            report_arg = ast.List(elts=report_vals, ctx=ast.Load(), lineno=node.lineno, col_offset=0)
            report_arg = ast.fix_missing_locations(report_arg)

            func_args = [result_lambda, arg_dict, keys_arg, report_arg]

            call = ast.Call(
                func=attr, args=func_args, keywords=ccollector.things, starargs=None, kwargs=None, lineno=node.lineno)

            node.test = call
            ast.fix_missing_locations(call)
            for i in ccollector.things:
                ast.fix_missing_locations(i)
            ast.fix_missing_locations(result_lambda)
            ast.fix_missing_locations(keys_arg)
            ast.fix_missing_locations(report_arg)



        return node


def get_names(node):
    ret_val = []
    for i in ast.walk(node):
        if isinstance(i, ast.Name):
            if i.id.startswith('cmp_') or i.id.startswith('thresh_') or i.id.startswith('value_') or i.id.startswith(
                    'res_'):
                ret_val.append(i)
            else:
                print 'ERROR unexpected name!', i.id
    return ret_val


class ComparisionInfo(object):
    """Hold the names of a comparision"""

    def __init__(self, key, comp, thresh, res):
        self.key = key
        self.comp = comp
        self.thresh = thresh
        self.res = res


class ComparisionCollector(ast.NodeTransformer):
    """Class to transform the runtime into something
        that instead of doing calculations pulls them out into all
        other

    """

    def __init__(self, thresholds, lineno, keys):
        self.things = []
        self.thresholds = thresholds
        self.keys = keys
        self.lineno = lineno
        self.value_map = {}

        # keep track of things
        self.information = {'comp': [], 'thresh': [], 'res': [], 'vals': []}
        self.lambda_dic = {}

        # keep track of numberings
        self.cnum = 0
        self.tnum = 0
        self.rnum = 0
        self.vnum = 0
        self.lineno = lineno

    def create_thresh(self, node):
        n = 'thresh_{:d}'.format(self.tnum)
        self.tnum += 1
        keyword = ast.keyword(arg=n, value=node, lineno=self.lineno)
        self.information['thresh'].append(keyword)
        self.things.append(keyword)
        return keyword

    def create_comp(self, node):
        n = 'cmp_{:d}'.format(self.tnum)
        self.cnum += 1
        keyword = ast.keyword(arg=n, value=node, lineno=self.lineno)
        self.things.append(keyword)
        self.information['comp'].append(keyword)
        return keyword

    def create_val(self, node):
        n = 'value_{:d}'.format(self.vnum)
        self.vnum += 1
        keyword = ast.keyword(arg=n, value=node, lineno=self.lineno)
        self.things.append(keyword)
        self.information['vals'].append(keyword)
        return keyword

    def create_res(self, node):
        n = 'res_{:d}'.format(self.rnum)
        args = []
        names = []
        for i in get_names(node):
            args.append(ast.Name(id=i.id, ctx=ast.Param(), lineno=self.lineno))
            names.append(i.id)
        args = ast.arguments(args=args, vararg=None, kwarg=None, defaults=[])
        lv = ast.Lambda(args=args, body=node, lineno=self.lineno)
        self.rnum += 1
        keyword = ast.keyword(arg=n, value=lv, lineno=self.lineno)
        self.things.append(keyword)
        self.information['res'].append(keyword)
        self.lambda_dic[n] = names
        return keyword

    def visit_BoolOp(self, node):
        for idx, val in enumerate(node.values):
            # On boolean op if one of the values contains a threshold than we need to visit and transform that runtime
            if self.check_contains(val):
                new_node = self.visit(val)
                node.values[idx] = new_node
            # otherwise we can encapsulate the whole value in a name
            else:
                v = self.create_val(val)
                node.values[idx] = ast.Name(id=v.arg, ctx=ast.Load(), lineno=self.lineno)
        return node

    def visit_Compare(self, node):
        # Test to see if there is any part of the runtime that contains the thershold.  If not just replace the runtime
        if not self.check_contains(node):
            print 'Its not in there?'
            v = self.create_val(node)
            return ast.Name(id=v.arg, ctx=ast.Load(), lineno=self.lineno)

        # now we need to loop through and replace stuff
        key = None
        comp = None
        res = None
        thresh = None
        if node.left in self.thresholds:
            lookup = self.thresholds.index(node.left)
            key = self.keys[lookup]
            t = self.create_thresh(node.left)
            thresh = t
            node.left = ast.Name(id=t.arg, ctx=ast.Load(), lineno=self.lineno)
        elif self.check_contains(node.left):
            new_node = self.visit(node.left)
            node.left = new_node
        else:
            c = self.create_comp(node.left)
            comp = c
            node.left = ast.Name(id=c.arg, ctx=ast.Load(), lineno=self.lineno)
        for idx, val in enumerate(node.comparators):
            if val in self.thresholds:
                lookup = self.thresholds.index(val)
                key = self.keys[lookup]
                t = self.create_thresh(val)
                thresh = t
                node.comparators[idx] = ast.Name(id=t.arg, ctx=ast.Load(), lineno=self.lineno)
            elif self.check_contains(val):
                new_node = self.visit(val)
                node.comparators[idx] = new_node
            else:
                c = self.create_comp(node.left)
                comp = c
                node.comparators[idx] = ast.Name(id=c.arg, ctx=ast.Load(), lineno=self.lineno)
            if idx > 0:
                print "Multi comparitors. ERROR"
        r = self.create_res(node)
        res = r
        if thresh is None or comp is None or res is None:
            print 'Why is one None?!?!'
        else:
            self.value_map[key] = ComparisionInfo(key, comp, thresh, res)
        return ast.Name(id=r.arg, ctx=ast.Load(), lineno=self.lineno)

    def visit_UnaryOp(self, node):
        # TODO Fix this as well
        if node.operand in self.thresholds:
            lookup = self.thresholds.index(node.operand)
            key = self.keys[lookup]
            thresh = self.create_thresh(node.operand)
            comp = self.create_comp(node.operand)
            res = self.create_res(node.operand)
            self.value_map[key] = ComparisionInfo(key, comp, thresh, res)



            node.operand = ast.Name(id=res.arg, ctx=ast.Load(), lineno=self.lineno)
        elif self.check_contains(node.operand):
            node.operand = self.visit(node.operand)
        return node

    def check_contains(self, node):
        v = NodeFinder(self.thresholds)
        v.visit(node)
        return v.found


class NodeFinder(ast.NodeVisitor):
    def __init__(self, targets):
        self.targets = targets
        self.found = False

    def generic_visit(self, node):
        if node in self.targets:
            self.found = True
        else:
            ast.NodeVisitor.generic_visit(self, node)


def get_thresh_rep(attr, start=str()):
    """get the name recursively defined"""
    if isinstance(attr, ast.Name):
        name = attr.id
    elif isinstance(attr, ast.Attribute):
        name = get_thresh_rep(attr.value, start) + '.' + get_thresh_rep(attr.attr, start)
    elif isinstance(attr, str):
        name = attr
    elif isinstance(attr, ast.Num):
        name = str(attr.n)
    elif isinstance(attr, int):
        return str(attr)
    elif isinstance(attr, float):
        return str(attr)
    else:
        name = ''
    return name
