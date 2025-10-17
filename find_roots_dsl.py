import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from frontend.parser import parse_input
from frontend.ast import VarNode, NumNode, BinOpNode
from codegen.cuda_codegen import generate_roots_kernel

def find_polynomial_roots(expr):
    """Find roots of a polynomial specified in DSL format."""
    try:
        # Parse the expression
        ast = parse_input(expr)
        
        # Generate kernel code
        kernel_code = generate_roots_kernel(ast)
        
        # Get degree from the expression
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
        
        # Allocate device memory for roots
        roots_real = np.zeros(degree, dtype=np.float32)
        roots_imag = np.zeros(degree, dtype=np.float32)
        
        # Allocate device memory
        roots_real_gpu = cuda.mem_alloc(roots_real.nbytes)
        roots_imag_gpu = cuda.mem_alloc(roots_imag.nbytes)
        
        # Initialize device memory
        cuda.memcpy_htod(roots_real_gpu, roots_real)
        cuda.memcpy_htod(roots_imag_gpu, roots_imag)
        
        # Compile and launch kernel
        mod = SourceModule(kernel_code)
        kernel = mod.get_function("find_roots_kernel")
        
        # Launch kernel with a single thread since we only need one
        kernel(
            roots_real_gpu,
            roots_imag_gpu,
            np.int32(degree),
            block=(1, 1, 1),
            grid=(1, 1)
        )
        
        # Copy results back
        cuda.memcpy_dtoh(roots_real, roots_real_gpu)
        cuda.memcpy_dtoh(roots_imag, roots_imag_gpu)
        
        # Free device memory
        roots_real_gpu.free()
        roots_imag_gpu.free()
        
        return roots_real, roots_imag
        
    except Exception as e:
        print(f"Error finding roots: {str(e)}")
        raise

def main():
    # Example polynomials in DSL format
    polynomials = [
        "y = x * x + 2 * x + 1",  # x^2 + 2x + 1
        "y = x * x * x + 2 * x * x + 3 * x + 4",  # x^3 + 2x^2 + 3x + 4
        "y = x * x * x * x + 2 * x * x * x + 3 * x * x + 4 * x + 5"  # x^4 + 2x^3 + 3x^2 + 4x + 5
    ]
    
    for i, expr in enumerate(polynomials):
        print(f"\nPolynomial {i+1}: {expr}")
        roots_real, roots_imag = find_polynomial_roots(expr)
        print("Roots:")
        for j in range(len(roots_real)):
            print(f"  Root {j+1}: {roots_real[j]:.6f} + {roots_imag[j]:.6f}i")

if __name__ == "__main__":
    main() 