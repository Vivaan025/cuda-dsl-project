# def generate_cuda_kernel(ast, vector_size=1024):
#     # Only supports: out = a + b, out = a * b, out = a + b * c, etc.
#     # Assumes all variables are vectors of length vector_size.
#     kernel_name = "kernel"
#     var_names = set()

#     def expr_to_cuda(node):
#         if hasattr(node, 'name'):
#             var_names.add(node.name)
#         if node.__class__.__name__ == 'VarNode':
#             return f"{node.name}[i]"
#         elif node.__class__.__name__ == 'NumNode':
#             return str(node.value)
#         elif node.__class__.__name__ == 'BinOpNode':
#             left = expr_to_cuda(node.left)
#             right = expr_to_cuda(node.right)
#             return f"({left} {node.op} {right})"
#         else:
#             raise NotImplementedError(f"Unknown node: {node}")

#     out_var = ast.var_name
#     expr_code = expr_to_cuda(ast.expr)
#     input_vars = [v for v in var_names if v != out_var]

#     # Generate CUDA kernel
#     kernel = f"""\
# __global__ void {kernel_name}(float* {out_var}, {', '.join(['float* ' + v for v in input_vars])}, int n) {{
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if(i < n) {{
#         {out_var}[i] = {expr_code};
#     }}
# }}
# """
#     return kernel
# codegen/cuda_codegen.py

from frontend.ast import BinOpNode, VarNode, NumNode
from jinja2 import Environment, FileSystemLoader
import os

def _expr_to_cuda(node, idx='i'):
    if isinstance(node, VarNode):
        return f"{node.name}[{idx}]"
    elif isinstance(node, NumNode):
        return str(node.value)
    elif isinstance(node, BinOpNode):
        left = _expr_to_cuda(node.left, idx)
        right = _expr_to_cuda(node.right, idx)
        if node.op == '@':
            raise ValueError("Use generate_matmul_kernel for matrix multiplication")
        return f"({left} {node.op} {right})"
    else:
        raise NotImplementedError(f"Unknown node: {node}")

def get_template_env():
    templates_dir = os.path.join(os.path.dirname(__file__), '../templates')
    return Environment(loader=FileSystemLoader(templates_dir))

def generate_elementwise_kernel(ast, kernel_name="kernel"):
    out_var = ast.var_name
    expr_code = _expr_to_cuda(ast.expr)
    input_vars = set()

    def collect_vars(node):
        if isinstance(node, VarNode):
            input_vars.add(node.name)
        elif isinstance(node, BinOpNode):
            collect_vars(node.left)
            collect_vars(node.right)
    collect_vars(ast.expr)
    input_vars.discard(out_var)
    params = [out_var] + list(input_vars)

    env = get_template_env()
    template = env.get_template('elementwise.cu.j2')
    return template.render(
        kernel_name=kernel_name,
        args=params,
        output_var=out_var,
        expression=expr_code
    )

def generate_matmul_kernel(ast, kernel_name="kernel"):
    out_var = ast.var_name
    left = ast.expr.left.name
    right = ast.expr.right.name

    env = get_template_env()
    template = env.get_template('matmul.cu.j2')
    return template.render(
        kernel_name=kernel_name,
        output_var=out_var,
        left_var=left,
        right_var=right
    )

def generate_kernel(ast, kernel_name="kernel"):
    if isinstance(ast.expr, BinOpNode) and ast.expr.op == '@':
        return generate_matmul_kernel(ast, kernel_name=kernel_name)
    else:
        return generate_elementwise_kernel(ast, kernel_name=kernel_name)
