from docutils.nodes import table
import sys

__author__ = 'ataylor'


import inspect
import ast_tools
import ast
from collections import defaultdict

from backward_analysis import BasicVisitor, get_local_pub_srv



class ObjectOutsideMap(object):
    """" This object contains a matp that maps variables to outside
    modules/classes and than it maps those modules/classes to functions which
    contain publish calls"""

    def __init__(self):
        self.variable_map = dict()
        self.function_map = defaultdict(set)
        self.total_map = defaultdict(set)

    def contains_pub(self, variable, function):
        pass

    def get_functions(self, cls):
        pass

    def print_out(self):
        for k,v in self.total_map.iteritems():
            print k
            for func in list(v):
                print '\t', func

    def populate_map(self, variable, module, cls):
        src_code = get_code_from_pkg_class(module, cls)
        tree = ast.parse(src_code)
        vals = get_local_pub_srv(tree)
        for i in vals:
            if i.cls.name == cls:
                for var in variable:
                    self.variable_map[var] = cls
                    self.function_map[cls].add(i.func.name)
                    self.total_map[var].add(i.func.name)

def get_call_objects(node, import_names):
    nn = ast_tools.get_name(node)
    for i in import_names:
        if nn.startswith(i[1]):
            obj = nn[nn.index(i[1])+len(i[1])+1:]
            return i[2], obj
    return None, None


class ImportFinder(ast.NodeVisitor):

    def __init__(self):
        # Names is a list of tuples. First element module_names, second names in file, third element is module
        self.names = []

    def visit_Import(self, node):
        for i in node.names:
            if i.asname is None:
                self.names.append((i.name, i.name, i.name))
            else:
                self.names.append((i.name, i.asname, i.name))

    def visit_ImportFrom(self, node):
        for i in node.names:
            if i.asname is None:
                self.names.append((i.name, i.name, node.module))
            else:
                self.names.append((i.name, i.asname, node.module))


class OutsideChecker(ast.NodeVisitor):

    def __init__(self, names, src_code):
        self.names = names
        self.src_code = src_code
        self.objects = []
        self.outside_class_map = ObjectOutsideMap()

    def visit_Assign(self, node):
        #if it is a call check to see if it came from imports
        if isinstance(node.value, ast.Call):
            t = [ast_tools.get_name(x) for x in node.targets]
            if isinstance(node.value.func, ast.Attribute):
                mod, obj = get_call_objects(node.value.func, self.names)
                if mod is not None:
                    obj_type = get_obj_type(mod, obj)
                    if obj_type == 'class':
                        # store the class type here and map all of the functions that contain outside  values
                        self.outside_class_map.populate_map(t, mod, obj)




            elif isinstance(node.value.func, ast.Name):
                #TODO: handle pure names
                pass

        self.generic_visit(node)

    # def visit_Call(self, node):
    #     self.generic_visit(node)
    #     if self.match:
    #         ast_tools.print_code_node(node, self.src_code)
    #         # get_obj_type(self.matched_name[2], self.matched_name[0])
    #         print '\n'
    #         self.match = False
    #         self.matched_obj = None


def get_src_code(cls_obj):
    src_file = inspect.getsourcefile(cls_obj)
    with open(src_file) as f:
        return f.read()


def get_code_from_pkg_class(package, cls):
    val = __import__(package)
    attr = getattr(val, cls)
    return get_src_code(attr)


def get_obj_type(package, name):
    print package, name
    pkg = __import__(package)
    obj = getattr(pkg, name)
    if inspect.isclass(obj):
        return 'class'
    elif inspect.isfunction(obj):
        return 'function'
    else:
        return 'unkown'


def build_import_list(tree=None, fname=None, src_code=None):
    if tree is None and fname is None:
        print("Error no file or tree")
        return None
    if tree is None:
        src_code = open(fname).read()
        tree = ast.parse(src_code)
        src_code = src_code.split('\n')

    import_finder = ImportFinder()
    import_finder.visit(tree)
    oc = OutsideChecker(import_finder.names, src_code)
    oc.visit(tree)
    oc.outside_class_map.print_out()


if __name__ == '__main__':
    fname = sys.argv[1]
    build_import_list(fname=fname)


    # package = 'baxter_interface'
    # cls = 'Head'
    # for i in get_code_from_pkg_class(package, cls):
    #     print i


