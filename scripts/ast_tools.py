import ast


def get_node_code(node, code):
    return code[node.lineno - 1].lstrip().rstrip()


def print_code_node(node, code):
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
        print node
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
