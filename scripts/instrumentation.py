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
    ast.fix_missing_locations(tree)
    code = compile(tree, fname, mode='exec')
    sys.argv = [fname] +  args
    ns = {'__name__': '__main__'}
    # exec (code, ns)


def add_import_statement(node):
    """Add the import statement to the top of the code before anything is done"""
    new_node = ast.Import(names=[ast.alias(name='reporting', asname=None)])
    new_node = ast.copy_location(new_node, node.body[0])

    ast.increment_lineno(node.body[0], 1)
    node.body = [new_node] + node.body
    return node


class InstrumentVisitor(ast.NodeTransformer):

    def __init__(self, thresholds, fname, code, verbose):
        self.fname = fname
        self.tmap = {}
        self.code = code
        self.verbose = verbose
        for i in thresholds:
            self.tmap[i] = thresholds[i]
        instrumented = set()


    def visit_If(self, node):
        self.handle_node(node)
        self.generic_visit(node)
        return node


    def visit_While(self, node):
        self.handle_node(node)
        self.generic_visit(node)
        return node

    def handle_node(self, node):
        if node in self.tmap:
            if isinstance(node.test, ast.Compare):
                print 'compare'
            elif isinstance(node.test, ast.BoolOp):
                print 'bool'
            elif isinstance(node.test, ast.UnaryOp):
                print 'unary'
            else:
                print 'Unexpected type here'
                assert False
            ccollector = ComparisionCollector(self.tmap[node])
            val = ccollector.visit(node.test)
            print val
            print ast.dump(val)
            for i in ccollector.things:
                print '\t', ast.dump(i)

            print ccollector.things


class ComparisionCollector(ast.NodeTransformer):

    def __init__(self, thresholds):
        self.things = []
        self.thresholds = thresholds
        self.cnum = 0
        self.tnum = 0
        self.rnum = 0
        self.vnum = 0


    def visit_BoolOp(self, node):
        for idx, val in  enumerate(node.values):
            print idx, val
            child_list = [x for x in ast.walk(val)]
            for j in self.thresholds:
                if j in child_list:
                    new_node = self.visit(val)
                    print '\t', new_node
                    node.values[idx] = new_node
                    break
            else:
                n = 'value_{:d}'.format(self.vnum)
                self.vnum += 1
                keyword = ast.keyword(arg=n, value=val)
                self.things.append(keyword)
                node.values[idx] = ast.Name(id=n, ctx=ast.Load())
            print '\n\n'
        return node



    def visit_Compare(self, node):
        thresh = []
        others = []
        if node.left in self.thresholds:
            thresh.append(node.left)
        else:
            others.append(node.left)
        for i in node.comparators:
            if i in self.thresholds:
                thresh.append(i)
            else:
                others.append(i)
        # Add this comparision to the comparitors

        for idx, val in enumerate(thresh):
            n = 'thersh_{:d}'.format(self.tnum)
            self.tnum += 1
            keyword = ast.keyword(arg=n, value=val)
            self.things.append(keyword)

        for idx, val in enumerate(others):
            n = 'cmp_{:d}'.format(self.tnum)
            self.tnum += 1
            keyword = ast.keyword(arg=n, value=val)
            self.things.append(keyword)

        n = 'res_{:d}'.format(self.rnum)
        keyword = ast.keyword(arg=n, value=node)
        self.things.append(keyword)
        self.generic_visit(node)
        return ast.Name(id=n, ctx=ast.Load())


    def visit_UnaryOp(self, node):
        #TODO Fix this as well
        values = [x for x in ast.iter_child_nodes(node.operand)]
        if node.operand in self.thresholds:
            n = 'thersh_{:d}'.format(self.tnum)
            self.tnum += 1
            keyword = ast.keyword(arg=n, value=node.operand)
            self.things.append(keyword)
            n = 'cmp_{:d}'.format(self.tnum)
            self.tnum += 1
            keyword = ast.keyword(arg=n, value=node.operand)
            self.things.append(keyword)
            n = 'res_{:d}'.format(self.rnum)
            keyword = ast.keyword(arg=n, value=node)
            node.operand = ast.Name(id=n, ctx=ast.Load())
        else:
            for i in self.thresholds:
                if i in values:
                    node.operand = self.visit(node.operand)
        return node





#Old call are below here!
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
    sys.argv = [fname] +  args
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
