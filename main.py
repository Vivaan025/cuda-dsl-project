# from frontend.parser import parse_input, print_ast
# from optimizer.optimize import optimize
# from codegen.cuda_codegen import generate_cuda_kernel
# from runtime.runtime import simulate_vector_add

# def main():
#     expr = "out = a + b * 2"
#     ast = parse_input(expr)
#     print("=== AST ===")
#     print_ast(ast)
#     print("\n=== Optimized AST  ===")
#     ast = optimize(ast)
#     print_ast(ast)
#     print("\n=== Generated CUDA Kernel ===")
#     kernel_code = generate_cuda_kernel(ast)
#     print(kernel_code)
#     print("\n=== Simulated Runtime Output ===")
#     a = [1, 2, 3]
#     b = [4, 5, 6]
#     out = simulate_vector_add(a, b)
#     print(f"a = {a}")
#     print(f"b = {b}")
#     print(f"out = {out}")

# if __name__ == "__main__":
#     main()
# main.py

from tests.test_kernels import test_all

if __name__ == "__main__":
    test_all()
    print("All tests passed.")