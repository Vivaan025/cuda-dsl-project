#!/usr/bin/env python3
"""
Direct Python testing examples
Copy these snippets into a Python interpreter or script
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

# Example 1: Test basic vector operations
def test_vector_operations():
    from frontend.parser import parse_input
    from codegen.cuda_codegen import generate_kernel
    from tests.test_kernels import run_vector_kernel
    
    # Test data
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    
    # Test expression
    expr = "out = a + b * 2"
    
    # Parse and generate
    ast = parse_input(expr)
    kernel_code = generate_kernel(ast)
    
    # Run on GPU
    result = run_vector_kernel(expr, a, b)
    
    # Compare with CPU
    cpu_result = a + b * 2
    error = np.max(np.abs(result - cpu_result))
    
    print(f"Expression: {expr}")
    print(f"GPU result: {result}")
    print(f"CPU result: {cpu_result}")
    print(f"Error: {error}")

# Example 2: Test polynomial root finding
def test_polynomial():
    from find_roots_dsl import find_polynomial_roots
    
    expr = "y = x * x - 5 * x + 6"  # Roots should be 2 and 3
    roots_real, roots_imag = find_polynomial_roots(expr)
    
    print(f"Polynomial: {expr}")
    print(f"Roots: {roots_real} + {roots_imag}i")

# Example 3: Test AST parsing
def test_parsing():
    from frontend.parser import parse_input, print_ast
    
    expressions = [
        "out = a + b",
        "y = x * x + 2 * x + 1", 
        "result = 3 * a - b / 2"
    ]
    
    for expr in expressions:
        print(f"\nExpression: {expr}")
        ast = parse_input(expr)
        print_ast(ast)

# Example 4: Test code generation only
def test_codegen():
    from frontend.parser import parse_input
    from codegen.cuda_codegen import generate_kernel
    
    expr = "out = a * b + 2"
    ast = parse_input(expr)
    kernel_code = generate_kernel(ast)
    
    print(f"Expression: {expr}")
    print("Generated CUDA code:")
    print("-" * 40)
    print(kernel_code)

if __name__ == "__main__":
    print("Running direct Python tests...")
    
    print("\n=== Vector Operations ===")
    test_vector_operations()
    
    print("\n=== Polynomial Roots ===")
    test_polynomial()
    
    print("\n=== AST Parsing ===")
    test_parsing()
    
    print("\n=== Code Generation ===")
    test_codegen()