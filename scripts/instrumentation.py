import ast
import pprinter
import sys

from ast_tools import get_string_repr


# In this module I have botht the older version of instrumentation and the new version of isntrumentation.
# The old version is the replace values series of calls and is at the bottom of the file.as
# The new version is at the top and only exports select information that just reports threshold values
# and comparision values in the code

def instrument_thresholds(tree, thresholds, fname, code, verbose, args):
    tree = InstrumentVisitor(thresholds, fname, code, verbose).visit(tree)
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

    def __init__(self, thresholds, fname, code, verbose):
        """setup"""
        self.fname = fname
        self.tmap = {}
        self.code = code
        self.verbose = verbose
        self.debug = False
        for i in thresholds:
            self.tmap[i] = thresholds[i]
        instrumented = set()

    def visit_If(self, node):
        """Just call the handle node function"""
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

            ccollector = ComparisionCollector(self.tmap[node], node.lineno)
            val = ccollector.visit(node.test)
            args = []
            names = []
            for i in get_names(val):
                args.append(ast.Name(id=i.id, ctx=ast.Param(), lineno=node.lineno))
                names.append(i.id)
            args = ast.arguments(args=args, vararg=None, kwarg=None, defaults=[])
            lv = ast.Lambda(args=args, body=val, lineno=node.lineno)
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
            dict_create = ast.Dict(keys=keys, values=values)
            # add the function call
            for i in ccollector.things:
                ast.fix_missing_locations(i)
            ast.fix_missing_locations(lv)
            name = ast.Name(id='reporting', ctx=ast.Load(lineno=node.lineno), lineno=node.lineno)
            attr = ast.Attribute(value=name, attr='report', ctx=ast.Load(lineno=node.lineno), lineno=node.lineno)

            func_args = [ast.Str(s=self.fname, lineno=node.lineno), ast.Num(n=node.lineno, lineno=node.lineno), lv,
                         dict_create]

            call = ast.Call(
                func=attr, args=func_args, keywords=ccollector.things, starargs=None, kwargs=None, lineno=node.lineno)

            node.test = call
            ast.fix_missing_locations(node)
            #now make the call

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


class ComparisionCollector(ast.NodeTransformer):
    """Class to transform the node into something
        that instead of doing calculations pulls them out into all
        other

    """

    def __init__(self, thresholds, lineno):
        self.things = []
        self.thresholds = thresholds
        self.lineno = lineno

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
            # On boolean op if one of the values contains a threshold than we need to visit and transform that node
            if self.check_contains(val):
                new_node = self.visit(val)
                node.values[idx] = new_node
            # otherwise we can encapsulate the whole value in a name
            else:
                v = self.create_val(val)
                node.values[idx] = ast.Name(id=v.arg, ctx=ast.Load(), lineno=self.lineno)
        return node

    def visit_Compare(self, node):
        # Test to see if there is any part of the node that contains the thershold.  If not just replace the node
        if not self.check_contains(node):
            print 'Its not in there?'
            v = self.create_val(node)
            return ast.Name(id=v.arg, ctx=ast.Load(), lineno=self.lineno)

        # now we need to loop through and replace stuff
        if node.left in self.thresholds:
            t = self.create_thresh(node.left)
            node.left = ast.Name(id=t.arg, ctx=ast.Load(), lineno=self.lineno)
        elif self.check_contains(node.left):
            new_node = self.visit(node.left)
            node.left = new_node
        else:
            c = self.create_comp(node.left)
            node.left = ast.Name(id=c.arg, ctx=ast.Load(), lineno=self.lineno)
        for idx, val in enumerate(node.comparators):
            if val in self.thresholds:
                t = self.create_thresh(val)
                node.comparators[idx] = ast.Name(id=t.arg, ctx=ast.Load(), lineno=self.lineno)
            elif self.check_contains(val):
                new_node = self.visit(val)
                node.comparators[idx] = new_node
            else:
                c = self.create_comp(node.left)
                node.comparators[idx] = ast.Name(id=c.arg, ctx=ast.Load(), lineno=self.lineno)
        r = self.create_res(node)
        return ast.Name(id=r.arg, ctx=ast.Load(), lineno=self.lineno)

    def visit_UnaryOp(self, node):
        # TODO Fix this as well
        if node.operand in self.thresholds:
            self.create_thresh(node.operand)
            self.create_comp(node.operand)
            res = self.create_res(node.operand)
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


# Old call are below here!
#
#
#
#
#
#

def replace_values(tree, thresholds, fname, code, verbose, args):
    """Replacement stuff to modify and exit the code."""
    tree = ModCalls(thresholds, fname, code, verbose).visit(tree)
    tree = add_import_statement(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, fname, mode='exec')
    sys.argv = [fname] + args
    ns = {'__name__': '__main__'}
    exec (code, ns)


class ModCalls(ast.NodeTransformer):
    def __init__(self, thresholds, fname, code, verbose):
        self.fname = fname
        self.tmap = {}
        self.code = code
        self.verbose = verbose
        for i in thresholds:
            self.tmap[i] = thresholds[i]

    def visit_If(self, node):
        if node in self.tmap:
            self.fix_constant_code(node)
        self.generic_visit(node)
        return node

    def visit_while(self, node):
        if node in self.tmap:
            self.fix_constant_code(node)
        self.generic_visit(node)
        return node

    def fix_constant_code(self, node):
        tstring = ':'.join((get_thresh_rep(x) for x in self.tmap[node]))

        line_code = self.code[node.lineno - 1].lstrip().strip()
        nav = NameAttrVisitor(self.fname, self.tmap[node])
        nav.visit(node.test)
        if nav.compare_number == 0:
            n = 'const_0_0'
            keyword = ast.keyword(arg=n, value=node.test)
            nav.things.append(keyword)
            n = 'cmp_0_0'
            keyword = ast.keyword(arg=n, value=node.test)
            nav.things.append(keyword)
            n = 'res_0'
            keyword = ast.keyword(arg=n, value=node.test)
            nav.things.append(keyword)

        for i in nav.things:
            ast.fix_missing_locations(i.value)

        name = ast.Name(id='reporting', ctx=ast.Load())
        attr = ast.Attribute(value=name, attr='report', ctx=ast.Load())
        func_args = [node.test, ast.Str(s=self.fname), ast.Str(
            s=str(node.lineno)), ast.Str(s=line_code), ast.Str(s=tstring)]

        call = ast.Call(
            func=attr, args=func_args, keywords=nav.things, starargs=None, kwargs=None)
        node.test = call


class NameAttrVisitor(ast.NodeVisitor):
    def __init__(self, name_pre, constants):
        self.things = []
        self.name_pre = name_pre
        self.constants = constants
        self.compare_number = 0

    def visit(self, node):
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

    def visit_Compare(self, node):
        const = []
        others = []
        if node.left in self.constants:
            const.append(node.left)
        else:
            others.append(node.left)
        for i in node.comparators:
            if i in self.constants:
                const.append(i)
            else:
                others.append(i)
        # Add this comparision to the comparitors
        for idx, val in enumerate(const):
            n = 'const_' + str(self.compare_number) + '_' + str(idx)
            keyword = ast.keyword(arg=n, value=val)
            self.things.append(keyword)

        for idx, val in enumerate(others):
            n = 'cmp_' + str(self.compare_number) + '_' + str(idx)
            keyword = ast.keyword(arg=n, value=val)
            self.things.append(keyword)

        n = 'res_' + str(self.compare_number)
        keyword = ast.keyword(arg=n, value=node)
        self.things.append(keyword)
        self.compare_number += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        name = self.name_pre + \
               str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)


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
