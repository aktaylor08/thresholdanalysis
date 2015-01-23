import ast


def get_node_code(node, code):
    """Get code of a node"""
    return str(node.lineno) + ' ' + code[node.lineno - 1].lstrip().rstrip()


def print_code_node(node, code):
    """Print code of a node"""
    print get_node_code(node, code)


def get_name(attr, start=str()):
    """get the name recursively defined"""
    if isinstance(attr, ast.Name):
        name = attr.id
    elif isinstance(attr, ast.Attribute):
        name = get_name(attr.value, start) + '.' + get_name(attr.attr, start)
    elif isinstance(attr, str):
        name = attr
    else:
        name = ''
    return name


def get_node_names(node):
    """Get names involved with a node"""
    nv = NameVisitor()
    nv.visit(node)
    return nv.names


def get_node_variables(node):
    """Get the variables involved within a node
    By this we mean the rhs of assignments, tests in if statements,
    and all of that stuff"""
    names = []
    if isinstance(node, ast.If):
        cur_names = get_node_names(node.test)
        for i in cur_names:
            names.append(i)
    elif isinstance(node, ast.While):
        cur_names = get_node_names(node.test)
        for i in cur_names:
            names.append(i)
    elif isinstance(node, ast.Call):
        for arg in node.args:
            cur_names = get_node_names(arg)
            for i in cur_names:
                names.append(i)
    elif isinstance(node, ast.Assign):
        cur_names = get_node_names(node.value)
        for i in cur_names:
            names.append(i)
    elif isinstance(node, ast.AugAssign):
        cur_names = get_node_names(node.value)
        for i in cur_names:
            names.append(i)
        cur_names = get_node_names(node.target)
        for i in cur_names:
            names.append(i)

    elif isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Call):
            for i in node.value.args:
                cur_names = get_node_names(i)
                for j in cur_names:
                    names.append(j)
            for i in node.value.keywords:
                cur_names = get_node_names(i.value)
                for j in cur_names:
                    names.append(j)
                    # skipping STARARGS AND KWARGS for now
        else:
            print "UNKNOWN EXPRESSION. DON'T KNOW WHAT TO DO"
    else:
        print "Unsupported type of node....", type(node)
    return names


def get_string_repr(node):
    if isinstance(node, ast.Name):
        return get_name(node)

    if isinstance(node, ast.Attribute):
        return get_name(node)

    if isinstance(node, ast.UnaryOp):
        op = node.op
        if isinstance(op, ast.UAdd):
            return '+ ' + get_string_repr(node.operand)
        if isinstance(op, ast.USub):
            return '- ' + get_string_repr(node.operand)
        if isinstance(op, ast.Not):
            return 'not ' + get_string_repr(node.operand)
        if isinstance(op, ast.Invert):
            return '~ ' + get_string_repr(node.operand)

    if isinstance(node, ast.Subscript):
        print node.slice
        val = get_string_repr(node.value)
        slc = get_string_repr(node.slice)
        return val + '[' + slc + ']'

    if isinstance(node, ast.Index):
        return get_string_repr(node.value)

    if isinstance(node, ast.Slice):
        val = get_string_repr(node.lower) + ":" + get_string_repr(node.upper)
        if node.step is not None:
            val = val + ":" + get_string_repr(node.step)
        return val

    if isinstance(node, ast.ExtSlice):
        vals = ', '.join([get_string_repr(x) for x in node.dims])
        return vals

    if isinstance(node, ast.BinOp):
        op = node.op
        left = node.left
        right = node.right
        if isinstance(op, ast.Add):
            return get_string_repr(left) + ' + ' + get_string_repr(right)
        if isinstance(op, ast.Sub):
            return get_string_repr(left) + ' - ' + get_string_repr(right)
        if isinstance(op, ast.Mult):
            return get_string_repr(left) + ' * ' + get_string_repr(right)
        if isinstance(op, ast.Div):
            return get_string_repr(left) + ' / ' + get_string_repr(right)
        if isinstance(op, ast.FloorDiv):
            return get_string_repr(left) + ' // ' + get_string_repr(right)
        if isinstance(op, ast.Mod):
            return get_string_repr(left) + ' % ' + get_string_repr(right)
        if isinstance(op, ast.Pow):
            return get_string_repr(left) + ' ** ' + get_string_repr(right)
        if isinstance(op, ast.LShift):
            return get_string_repr(left) + ' << ' + get_string_repr(right)
        if isinstance(op, ast.RShift):
            return get_string_repr(left) + ' >> ' + get_string_repr(right)
        if isinstance(op, ast.BitOr):
            return get_string_repr(left) + ' | ' + get_string_repr(right)
        if isinstance(op, ast.BitXor):
            return get_string_repr(left) + ' ^ ' + get_string_repr(right)
        if isinstance(op, ast.BitAnd):
            return get_string_repr(left) + ' & ' + get_string_repr(right)

    if isinstance(node, ast.BoolOp):
        op = node.op
        if isinstance(op, ast.Or):
            return ' or '.join([get_string_repr(x) for x in node.values])
        if isinstance(op, ast.And):
            return ' and '.join([get_string_repr(x) for x in node.values])

    if isinstance(node, ast.Compare):
        string = get_string_repr(node.left)
        for op, val in zip(node.ops, node.comparators):
            string += ' ' + get_string_repr(op) + ' ' + get_string_repr(val)
        return string

    if isinstance(node, ast.Call):
        string = get_string_repr(node.func)
        return string + '( )'

    if isinstance(node, ast.Eq):
        return ' == '
    if isinstance(node, ast.NotEq):
        return ' != '
    if isinstance(node, ast.Lt):
        return ' < '
    if isinstance(node, ast.LtE):
        return ' <= '
    if isinstance(node, ast.Gt):
        return ' > '
    if isinstance(node, ast.GtE):
        return ' >= '
    if isinstance(node, ast.Is):
        return ' is '
    if isinstance(node, ast.IsNot):
        return ' is not '
    if isinstance(node, ast.In):
        return ' in '
    if isinstance(node, ast.NotIn):
        return ' not in '

    if isinstance(node, ast.Num):
        return str(node.n)

    if isinstance(node, ast.FunctionDef):
        return 'def ' + str(node.name) + '(...):'

    if isinstance(node, ast.Str):
        return node.s


class NameVisitor(ast.NodeVisitor):
    """Super simple visitor to get the the names in a node"""

    def __init__(self):
        self.names = []

    def visit_Attribute(self, node):
        self.names.append(get_name(node))

    def visit_Name(self, node):
        self.names.append(get_name(node))

    def visit_Call(self, node):
        # Skip function name but go on to everything else
        for i in node.args:
            self.generic_visit(i)
        for i in node.keywords:
            self.generic_visit(i)

        if node.starargs is not None:
            if isinstance(node.starargs, list):
                for i in node.starargs:
                    self.generic_visit(i)
            else:
                self.generic_visit(node.starargs)

        if node.kwargs is not None:
            if isinstance(node.kwargs, list):
                for i in node.kwargs:
                    self.generic_visit(i)
            else:
                self.generic_visit(node.kwargs)
            self.generic_visit(node.starargs)


class ContainingVisitor(ast.NodeVisitor):
    """finds if the program is part of an if statement"""

    def __init__(self, target):
        self.target = target
        self.res = None
        self.found = False
        self.depth = 0
        self.parent = None

    def visit_FunctionDef(self, node):
        self.depth += 1
        op = self.parent
        self.parent = node
        if not self.found:
            self.generic_visit(node)
        self.parent = op
        self.depth -= 1

    def visit_If(self, node):
        self.depth += 1
        op = self.parent
        self.parent = node
        if not self.found:
            self.generic_visit(node)
        self.depth -= 1
        self.parent = op

    def visit_While(self, node):
        self.depth += 1
        op = self.parent
        self.parent = node
        if not self.found:
            self.generic_visit(node)
        self.depth -= 1
        self.parent = op

    def generic_visit(self, node):
        if node == self.target:
            self.found = True
            self.res = self.parent
        ast.NodeVisitor.generic_visit(self, node)