import sys
import inspect
import ast
from collections import defaultdict

import ast_tools
from backward_analysis import BasicVisitor, ServiceFinderVisitor, ServiceCallFinder, TreeObject


class OutsidePubFinder(BasicVisitor):
    def __init__(self):
        BasicVisitor.__init__(self)
        self.publish_calls = []
        self.in_expr = False
        self.found = False

    def visit_Expr(self, node):
        self.current_expr = node
        self.in_expr = True
        self.generic_visit(node)
        if self.found:
            self.publish_calls.append(
                TreeObject(self.current_class,
                           self.current_function, self.current_expr, node))
            self.found = False
        self.in_expr = False
        self.current_expr = None

    def visit_Attribute(self, node):
        if node.attr == 'publish':
            self.found = True


def get_outside_pub_svr(tree):
    service_finder = ServiceFinderVisitor()
    service_finder.visit(tree)
    call_finder = ServiceCallFinder(service_finder.proxies)
    call_finder.visit(tree)
    opf = OutsidePubFinder()
    opf.visit(tree)
    return call_finder.calls + opf.publish_calls


class ObjectOutsideMap(object):
    """" This object contains a matp that maps variables to outside
    modules/classes and than it maps those modules/classes to functions which
    contain publish calls"""

    def __init__(self):
        self.variable_map = dict()
        self.function_map = defaultdict(set)
        self.total_map = defaultdict(set)

    def outside(self, call):
        """Check to see if the passed node calls an outside function"""
        if isinstance(call.func, ast.Attribute):
            name = ast_tools.get_name(call.func.value)
            if call.func.attr in self.total_map[name]:
                return True
            else:
                return False
        else:
            print '\t', call.lineno, ast.dump(call)

    def get_functions(self, cls):
        return self.function_map[cls]

    def print_out(self):
        for k, v in self.total_map.iteritems():
            print k
            for func in list(v):
                print '\t', func

    def populate_map(self, variable, module, cls):
        """Populate the map with the variable name, the module, and any functions
        that publish within the class"""
        src_code = get_code_from_pkg_class(module, cls)
        tree = ast.parse(src_code)
        vals = get_outside_pub_svr(tree)
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
            obj = nn[nn.index(i[1]) + len(i[1]) + 1:]
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
        # if it is a call check to see if it came from imports
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
                # TODO: handle pure names
                pass

        self.generic_visit(node)


def get_src_code(cls_obj):
    src_file = inspect.getsourcefile(cls_obj)
    with open(src_file) as f:
        return f.read()


def get_code_from_pkg_class(package, cls):
    val = __import__(package)
    attr = getattr(val, cls)
    return get_src_code(attr)


def get_obj_type(package, name):
    pkg = __import__(package)
    obj = getattr(pkg, name)
    if inspect.isclass(obj):
        return 'class'
    elif inspect.isfunction(obj):
        return 'function'
    else:
        return 'unkown'


class OutsideCallFinder(BasicVisitor):

    def __init__(self, ocm):
        super(OutsideCallFinder, self).__init__()
        self.ocm = ocm
        self.outside_calls = []

    def visit_Call(self, node):
        is_pub = self.ocm.outside(node)
        if is_pub:
            print 'outside', node.lineno
            oc = TreeObject(self.current_class, self.current_function, self.current_expr, node)
            self.outside_calls.append(oc)





if __name__ == '__main__':
    fname = sys.argv[1]
    build_import_list(fname=fname)
