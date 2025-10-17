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
from ir.ast_to_ir import convert_ast_to_ir
from optimizer.optimize import optimize
from codegen.ir_to_cuda import generate_cuda_from_ir
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

def extract_polynomial_coeffs(node, degree):
    """Extract coefficients from a polynomial expression."""
    coeffs = [0.0] * (degree + 1)
    
    def get_term_info(node):
        """Get the power and coefficient of a term."""
        if isinstance(node, NumNode):
            return 0, float(node.value)  # power=0, coefficient=value
        elif isinstance(node, VarNode):
            return 1, 1.0  # power=1, coefficient=1
        elif isinstance(node, BinOpNode) and node.op == '*':
            left_power, left_coeff = get_term_info(node.left)
            right_power, right_coeff = get_term_info(node.right)
            return left_power + right_power, left_coeff * right_coeff
        else:
            return 0, 1.0  # Default case
    
    def extract_terms(node, sign=1):
        """Extract all terms from the polynomial."""
        if isinstance(node, BinOpNode):
            if node.op == '+':
                extract_terms(node.left, sign)
                extract_terms(node.right, sign)
            elif node.op == '-':
                extract_terms(node.left, sign)
                extract_terms(node.right, -sign)
            elif node.op == '*':
                power, coeff = get_term_info(node)
                if 0 <= power <= degree:
                    coeffs[power] += sign * coeff
        else:
            # Single term (number or variable)
            power, coeff = get_term_info(node)
            if 0 <= power <= degree:
                coeffs[power] += sign * coeff
    
    extract_terms(node)
    return coeffs

def polynomial_kernel(ast, kernel_name="kernel"):
    """
    Generate CUDA kernel for polynomial calculations.
    For degrees <= 3: Uses direct calculation
    For degrees > 3: Uses Durand-Kerner method
    """
    out_var = ast.var_name
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
    
    # Determine polynomial degree
    def get_degree(node):
        if isinstance(node, VarNode):
            return 1
        elif isinstance(node, NumNode):
            return 0
        elif isinstance(node, BinOpNode):
            if node.op == '*':
                return get_degree(node.left) + get_degree(node.right)
            elif node.op in ['+', '-']:
                return max(get_degree(node.left), get_degree(node.right))
            else:
                return 1
        return 0
    
    degree = get_degree(ast.expr)
    expr_code = _expr_to_cuda(ast.expr)
    
    # Extract coefficients
    coeffs = extract_polynomial_coeffs(ast.expr, degree)
    
    env = get_template_env()
    if degree <= 3:
        template = env.get_template('elementwise.cu.j2')
        return template.render(
            kernel_name=kernel_name,
            args=params,
            output_var=out_var,
            expression=expr_code
        )
    else:
        # For higher degrees, use Durand-Kerner method
        template = env.get_template('polynomial.cu.j2')
        return template.render(
            kernel_name=kernel_name,
            args=params,
            output_var=out_var,
            degree=degree,
            coeffs=coeffs
        )

def generate_trigonometric_kernel(ast, function_type, kernel_name="kernel"):
    """Generate CUDA kernel for trigonometric functions."""
    out_var = ast.var_name
    input_vars = set()
    
    def collect_vars(node):
        if isinstance(node, VarNode):
            input_vars.add(node.name)
        elif isinstance(node, BinOpNode):
            collect_vars(node.left)
            collect_vars(node.right)
    
    collect_vars(ast.expr)
    input_vars.discard(out_var)
    input_var = list(input_vars)[0] if input_vars else "x"
    
    env = get_template_env()
    template = env.get_template('trigonometric.cu.j2')
    return template.render(
        kernel_name=kernel_name,
        output_var=out_var,
        input_var=input_var,
        function_type=function_type
    )

def generate_differential_kernel(ast, method="central", kernel_name="kernel"):
    """Generate CUDA kernel for numerical differentiation."""
    out_var = ast.var_name
    input_vars = set()
    
    def collect_vars(node):
        if isinstance(node, VarNode):
            input_vars.add(node.name)
        elif isinstance(node, BinOpNode):
            collect_vars(node.left)
            collect_vars(node.right)
    
    collect_vars(ast.expr)
    input_vars.discard(out_var)
    input_var = list(input_vars)[0] if input_vars else "x"
    
    env = get_template_env()
    template = env.get_template('differential.cu.j2')
    return template.render(
        kernel_name=kernel_name,
        output_var=out_var,
        input_var=input_var,
        method=method
    )

def generate_kernel_optimized(ast, kernel_name="kernel", use_ir=True, **kwargs):
    """Optimized kernel generation using IR and optimization passes."""
    if use_ir:
        # Convert AST to IR
        ir_module = convert_ast_to_ir(ast, kernel_name)
        
        # Apply optimizations
        optimized_module = optimize(ir_module, target='cuda')
        
        # Generate CUDA code from optimized IR
        return generate_cuda_from_ir(optimized_module, kernel_name, optimized=True)
    else:
        # Fall back to original method
        return generate_kernel_legacy(ast, kernel_name, **kwargs)

def generate_kernel_legacy(ast, kernel_name="kernel", **kwargs):
    """Legacy kernel generation - kept for compatibility."""
    if isinstance(ast.expr, BinOpNode) and ast.expr.op == '@':
        return generate_matmul_kernel(ast, kernel_name=kernel_name)
    else:
        # Check for special function calls
        operation_type = kwargs.get('operation_type', 'auto')
        
        if operation_type in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh']:
            return generate_trigonometric_kernel(ast, operation_type, kernel_name)
        elif operation_type == 'diff':
            method = kwargs.get('method', 'central')
            return generate_differential_kernel(ast, method, kernel_name)
        else:
            # Check if this is a polynomial expression
            def is_polynomial(node):
                if isinstance(node, VarNode) or isinstance(node, NumNode):
                    return True
                elif isinstance(node, BinOpNode):
                    if node.op in ['+', '-', '*']:
                        return is_polynomial(node.left) and is_polynomial(node.right)
                    return False
                return False
            
            if is_polynomial(ast.expr):
                return polynomial_kernel(ast, kernel_name=kernel_name)
            else:
                return generate_elementwise_kernel(ast, kernel_name=kernel_name)

def generate_kernel(ast, kernel_name="kernel", **kwargs):
    """Main kernel generation dispatcher - uses legacy path by default for stability."""
    use_optimization = kwargs.get('optimize', False)  # Changed default to False
    
    if use_optimization:
        try:
            return generate_kernel_optimized(ast, kernel_name, **kwargs)
        except Exception as e:
            # Fall back to legacy if optimization fails
            print(f"Optimization failed: {e}. Falling back to legacy generation.")
            return generate_kernel_legacy(ast, kernel_name, **kwargs)
    else:
        return generate_kernel_legacy(ast, kernel_name, **kwargs)

def generate_roots_kernel(ast, kernel_name="find_roots_kernel"):
    """
    Generate CUDA kernel for finding roots of a polynomial expression.
    """
    # Determine polynomial degree
    def get_degree(node):
        if isinstance(node, VarNode):
            return 1
        elif isinstance(node, NumNode):
            return 0
        elif isinstance(node, BinOpNode):
            if node.op == '*':
                return get_degree(node.left) + get_degree(node.right)
            elif node.op in ['+', '-']:
                return max(get_degree(node.left), get_degree(node.right))
            else:
                return 1
        return 0
    
    degree = get_degree(ast.expr)
    
    # Extract coefficients
    coeffs = extract_polynomial_coeffs(ast.expr, degree)
    
    env = get_template_env()
    template = env.get_template('polynomial.cu.j2')
    return template.render(
        kernel_name=kernel_name,
        degree=degree,
        coeffs=coeffs
    )
