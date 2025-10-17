#!/usr/bin/env python3
"""
Interactive DSL Tester - Test your CUDA DSL expressions manually
Usage: python interactive_dsl_tester.py
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.parser import parse_input
from codegen.cuda_codegen import generate_kernel
from tests.test_kernels import run_vector_kernel, run_matmul_kernel
from find_roots_dsl import find_polynomial_roots

def test_basic_expression():
    """Test basic arithmetic expressions."""
    print("=== Testing Basic Expressions ===")
    
    expressions = [
        "out = a + b",
        "out = a - b", 
        "out = a * b",
        "out = a / b",
        "out = a + b * 2",
        "out = 3 * a + 2 * b"
    ]
    
    # Create test data
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    
    for expr in expressions:
        print(f"\nExpression: {expr}")
        try:
            # Parse and generate kernel
            ast = parse_input(expr)
            print(f"✓ Parsed successfully")
            
            kernel_code = generate_kernel(ast)
            print(f"✓ Generated CUDA kernel ({len(kernel_code)} chars)")
            
            # Run on GPU
            result = run_vector_kernel(expr, a, b)
            print(f"✓ GPU Result: {result}")
            
            # Calculate CPU reference
            if "+" in expr and "*" in expr:
                if "3 * a + 2 * b" in expr:
                    cpu_result = 3 * a + 2 * b
                elif "a + b * 2" in expr:
                    cpu_result = a + b * 2
            elif "+" in expr:
                cpu_result = a + b
            elif "-" in expr:
                cpu_result = a - b
            elif "*" in expr:
                cpu_result = a * b
            elif "/" in expr:
                cpu_result = a / b
                
            error = np.max(np.abs(result - cpu_result))
            print(f"✓ CPU Reference: {cpu_result}")
            print(f"✓ Error: {error}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_polynomial_roots():
    """Test polynomial root finding."""
    print("\n=== Testing Polynomial Root Finding ===")
    
    polynomials = [
        ("Linear: y = 2 * x + 4", "y = 2 * x + 4", [-2.0]),
        ("Quadratic: y = x * x - 5 * x + 6", "y = x * x - 5 * x + 6", [2.0, 3.0]),
        ("Quadratic Complex: y = x * x + x + 1", "y = x * x + x + 1", "complex"),
        ("Cubic: y = x * x * x - 6 * x * x + 11 * x - 6", "y = x * x * x - 6 * x * x + 11 * x - 6", [1.0, 2.0, 3.0])
    ]
    
    for desc, expr, expected in polynomials:
        print(f"\n{desc}")
        print(f"Expression: {expr}")
        try:
            roots_real, roots_imag = find_polynomial_roots(expr)
            print(f"✓ Found {len(roots_real)} roots:")
            
            for i in range(len(roots_real)):
                if abs(roots_imag[i]) < 1e-6:
                    print(f"  Root {i+1}: {roots_real[i]:.6f}")
                else:
                    print(f"  Root {i+1}: {roots_real[i]:.6f} + {roots_imag[i]:.6f}i")
            
            if expected != "complex":
                real_roots = [roots_real[i] for i in range(len(roots_real)) if abs(roots_imag[i]) < 1e-6]
                real_roots.sort()
                expected.sort()
                print(f"✓ Expected: {expected}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_matrix_operations():
    """Test matrix operations."""
    print("\n=== Testing Matrix Operations ===")
    
    # Create test matrices
    A = np.random.randn(4, 3).astype(np.float32)
    B = np.random.randn(3, 5).astype(np.float32)
    
    expr = "C = A @ B"
    print(f"Expression: {expr}")
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    
    try:
        result = run_matmul_kernel(expr, A, B)
        cpu_result = np.dot(A, B)
        error = np.max(np.abs(result - cpu_result))
        
        print(f"✓ GPU Result shape: {result.shape}")
        print(f"✓ CPU Result shape: {cpu_result.shape}")
        print(f"✓ Max error: {error}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def interactive_mode():
    """Interactive mode for testing custom expressions."""
    print("\n=== Interactive Mode ===")
    print("Enter DSL expressions to test (type 'quit' to exit)")
    print("Examples:")
    print("  out = a + b")
    print("  y = x * x + 2 * x + 1")
    print("  C = A @ B")
    
    while True:
        try:
            expr = input("\nDSL> ").strip()
            if expr.lower() in ['quit', 'exit', 'q']:
                break
            
            if not expr:
                continue
                
            # Determine expression type
            if 'x' in expr and any(op in expr for op in ['*', '+', '-']):
                # Polynomial
                print("Detected: Polynomial expression")
                roots_real, roots_imag = find_polynomial_roots(expr)
                print(f"Found {len(roots_real)} roots:")
                for i in range(len(roots_real)):
                    if abs(roots_imag[i]) < 1e-6:
                        print(f"  Root {i+1}: {roots_real[i]:.6f}")
                    else:
                        print(f"  Root {i+1}: {roots_real[i]:.6f} + {roots_imag[i]:.6f}i")
                        
            elif '@' in expr:
                # Matrix multiplication
                print("Detected: Matrix operation")
                print("Using random 4x3 and 3x5 matrices for testing...")
                A = np.random.randn(4, 3).astype(np.float32)
                B = np.random.randn(3, 5).astype(np.float32)
                result = run_matmul_kernel(expr, A, B)
                print(f"Result shape: {result.shape}")
                print(f"Sample values: {result.flat[:5]}")
                
            else:
                # Vector operation
                print("Detected: Vector operation")
                print("Using test vectors [1,2,3,4] and [5,6,7,8]...")
                a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
                result = run_vector_kernel(expr, a, b)
                print(f"Result: {result}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("Goodbye!")

def main():
    """Main testing function."""
    print("CUDA DSL Interactive Tester")
    print("=" * 40)
    
    while True:
        print("\nChoose a testing mode:")
        print("1. Test basic expressions")
        print("2. Test polynomial root finding")
        print("3. Test matrix operations")
        print("4. Interactive mode (enter your own expressions)")
        print("5. Run all tests")
        print("6. Quit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                test_basic_expression()
            elif choice == '2':
                test_polynomial_roots()
            elif choice == '3':
                test_matrix_operations()
            elif choice == '4':
                interactive_mode()
            elif choice == '5':
                test_basic_expression()
                test_polynomial_roots()
                test_matrix_operations()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()