# frontend/parser.py

import re
import collections
from .ast import AssignNode, BinOpNode, VarNode, NumNode

def tokenize(expr):
    token_specification = [
        ('NUMBER',    r'\d+'),
        ('IDENT',     r'[a-zA-Z_][a-zA-Z_0-9]*'),
        ('ASSIGN',    r'='),
        ('PLUS',      r'\+'),
        ('MINUS',     r'-'),
        ('TIMES',     r'\*'),
        ('DIVIDE',    r'/'),
        ('LPAREN',    r'\('),
        ('RPAREN',    r'\)'),
        ('SKIP',      r'[ \t\n]+'),
        ('MISMATCH',  r'.'),
    ]
    tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
    for mo in re.finditer(tok_regex, expr):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise RuntimeError(f'Unexpected character {value}')
        yield (kind, value)

class TokenStream:
    def __init__(self, tokens):
        self.tokens = collections.deque(tokens)
    def peek(self):
        return self.tokens[0] if self.tokens else None
    def next(self):
        return self.tokens.popleft() if self.tokens else None

def parse_expr(ts):
    left = parse_term(ts)
    while ts.peek() and ts.peek()[0] in ['PLUS', 'MINUS']:
        op_token = ts.next()
        op = '+' if op_token[0] == 'PLUS' else '-'
        right = parse_term(ts)
        left = BinOpNode(left, op, right)
    return left

def parse_term(ts):
    left = parse_factor(ts)
    while ts.peek() and ts.peek()[0] in ['TIMES', 'DIVIDE']:
        op_token = ts.next()
        op = '*' if op_token[0] == 'TIMES' else '/'
        right = parse_factor(ts)
        left = BinOpNode(left, op, right)
    return left

def parse_factor(ts):
    token = ts.next()
    if token is None:
        raise SyntaxError("Unexpected end of input")
    kind, value = token
    if kind == 'NUMBER':
        return NumNode(int(value))
    elif kind == 'IDENT':
        return VarNode(value)
    elif kind == 'LPAREN':
        expr = parse_expr(ts)
        if ts.next()[0] != 'RPAREN':
            raise SyntaxError("Expected closing parenthesis")
        return expr
    else:
        raise SyntaxError(f"Unexpected token {value}")

def parse_assignment(ts):
    token = ts.next()
    if token is None or token[0] != 'IDENT':
        raise SyntaxError("Expected variable name for assignment")
    var_name = token[1]
    if ts.next()[0] != 'ASSIGN':
        raise SyntaxError("Expected '=' for assignment")
    expr = parse_expr(ts)
    return AssignNode(var_name, expr)

def parse_input(expression):
    tokens = list(tokenize(expression))
    ts = TokenStream(tokens)
    ast = parse_assignment(ts)
    if ts.peek() is not None:
        raise SyntaxError("Unexpected extra input after valid expression")
    return ast

def print_ast(node, level=0):
    indent = '  ' * level
    from .ast import AssignNode, BinOpNode, VarNode, NumNode
    if isinstance(node, AssignNode):
        print(f"{indent}Assign: {node.var_name} =")
        print_ast(node.expr, level + 1)
    elif isinstance(node, BinOpNode):
        print(f"{indent}Op: {node.op}")
        print_ast(node.left, level + 1)
        print_ast(node.right, level + 1)
    elif isinstance(node, VarNode):
        print(f"{indent}Var: {node.name}")
    elif isinstance(node, NumNode):
        print(f"{indent}Num: {node.value}")