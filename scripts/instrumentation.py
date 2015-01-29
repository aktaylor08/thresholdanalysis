import ast
import sys

from ast_tools import get_string_repr

def replace_values(tree, thresholds, fname, code, verbose, args):
    """Replacement stuff to modify and exit the code."""
    tree = ModCalls(thresholds, fname, code, verbose).visit(tree)
    tree = add_import_statement(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, fname, mode='exec')
    sys.argv = [fname] +  args
    ns = {'__name__': '__main__'}
    exec (code, ns)


def add_import_statement(node):
    """Add the import statement to the top of the code before anything is done"""
    new_node = ast.Import(names=[ast.alias(name='reporting', asname=None)])
    new_node = ast.copy_location(new_node, node.body[0])

    ast.increment_lineno(node.body[0], 1)
    node.body = [new_node] + node.body
    return node


class ModCalls(ast.NodeTransformer):
    def __init__(self, thresholds, fname, code, verbose):
        self.fname = fname
        self.tmap = {}
        self.code = code
        self.verbose = verbose
        for i in thresholds:
            self.tmap[i] = i

    def visit_If(self, node):
        if node in self.tmap:
            if self.verbose:
                print('modifying:', node.lineno, node)
            line_code = self.code[node.lineno - 1].lstrip().strip()
            nav = NameAttrVisitor(self.fname)
            nav.visit(node.test)
            for i in nav.things:
                ast.fix_missing_locations(i.value)

            name = ast.Name(id='reporting', ctx=ast.Load())
            attr = ast.Attribute(value=name, attr='report', ctx=ast.Load())
            func_args = [node.test, ast.Str(s=self.fname), ast.Str(
                s=str(node.lineno)), ast.Str(s=line_code)]

            call = ast.Call(
                func=attr, args=func_args, keywords=nav.things, starargs=None, kwargs=None)
            node.test = call
        self.generic_visit(node)
        return node



class NameAttrVisitor(ast.NodeVisitor):
    def __init__(self, name_pre):
        self.things = []
        self.name_pre = name_pre

    def visit(self, node):
        # print node
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

    def visit_Name(self, node):
        name = self.name_pre + \
               str(node.lineno) + ' value->' + get_string_repr(node)
        keyword = ast.keyword(arg=name, value=node)
        self.things.append(keyword)
