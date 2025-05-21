# codegen/cuda_codegen.py

def generate_cuda_kernel(ast, vector_size=1024):
    # Only supports: out = a + b, out = a * b, out = a + b * c, etc.
    # Assumes all variables are vectors of length vector_size.
    kernel_name = "kernel"
    var_names = set()

    def expr_to_cuda(node):
        if hasattr(node, 'name'):
            var_names.add(node.name)
        if node.__class__.__name__ == 'VarNode':
            return f"{node.name}[i]"
        elif node.__class__.__name__ == 'NumNode':
            return str(node.value)
        elif node.__class__.__name__ == 'BinOpNode':
            left = expr_to_cuda(node.left)
            right = expr_to_cuda(node.right)
            return f"({left} {node.op} {right})"
        else:
            raise NotImplementedError(f"Unknown node: {node}")

    out_var = ast.var_name
    expr_code = expr_to_cuda(ast.expr)
    input_vars = [v for v in var_names if v != out_var]

    # Generate CUDA kernel
    kernel = f"""\
__global__ void {kernel_name}(float* {out_var}, {', '.join(['float* ' + v for v in input_vars])}, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {{
        {out_var}[i] = {expr_code};
    }}
}}
"""
    return kernel
