# frontend/ast.py

class Node: pass

class AssignNode(Node):
    def __init__(self, var_name, expr):
        self.var_name = var_name
        self.expr = expr

class BinOpNode(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op  # '+', '-', '*', '/'
        self.right = right

class VarNode(Node):
    def __init__(self, name):
        self.name = name

class NumNode(Node):
    def __init__(self, value):
        self.value = value
