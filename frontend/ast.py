# frontend/ast.py
from typing import Any, List, Dict, Optional
from abc import ABC, abstractmethod

class Node(ABC):
    """Base AST node with visitor pattern support."""
    
    @abstractmethod
    def accept(self, visitor):
        pass
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_')]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

class AssignNode(Node):
    def __init__(self, var_name: str, expr: Node):
        self.var_name = var_name
        self.expr = expr
    
    def accept(self, visitor):
        return visitor.visit_assign(self)

class BinOpNode(Node):
    def __init__(self, left: Node, op: str, right: Node):
        self.left = left
        self.op = op  # '+', '-', '*', '/', '@'
        self.right = right
    
    def accept(self, visitor):
        return visitor.visit_binop(self)

class VarNode(Node):
    def __init__(self, name: str):
        self.name = name
    
    def accept(self, visitor):
        return visitor.visit_var(self)

class NumNode(Node):
    def __init__(self, value: float):
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_num(self)

# Enhanced AST nodes for better optimization
class UnaryOpNode(Node):
    def __init__(self, op: str, operand: Node):
        self.op = op  # '-', '+'
        self.operand = operand
    
    def accept(self, visitor):
        return visitor.visit_unary(self)

class CallNode(Node):
    """Function call node for built-in functions like sin, cos, etc."""
    def __init__(self, func_name: str, args: List[Node]):
        self.func_name = func_name
        self.args = args
    
    def accept(self, visitor):
        return visitor.visit_call(self)

class IndexNode(Node):
    """Array/vector indexing node."""
    def __init__(self, array: Node, index: Node):
        self.array = array
        self.index = index
    
    def accept(self, visitor):
        return visitor.visit_index(self)

