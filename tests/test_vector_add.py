# tests/test_vector_add.py

from frontend.parser import parse_input
from codegen.cuda_codegen import generate_cuda_kernel

def test_vector_add():
    expr = "out = a + b * 2"
    ast = parse_input(expr)
    kernel = generate_cuda_kernel(ast)
    assert "__global__ void kernel" in kernel
    assert "out[i] = (a[i] + (b[i] * 2));" in kernel

if __name__ == "__main__":
    test_vector_add()
    print("test_vector_add passed.")
