import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os
from frontend.parser import parse_input
from codegen.cuda_codegen import generate_kernel

def run_dsl_file(filename):
    # Read the DSL expression from file
    def get_abs_path(relative_path):
        # Get the directory where this script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, relative_path)

    with open(get_abs_path("examples/vector_add.dsl"), 'r') as f:
        expr = f.read().strip()
    ast = parse_input(expr)
    kernel_code = generate_kernel(ast)

    # Example: handle vector or matrix based on variable names
    if '@' in expr:
        # Matrix multiplication example
        M, N, P = 64, 128, 32
        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(N, P).astype(np.float32)
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C = np.empty((M, P), dtype=np.float32)
        C_gpu = cuda.mem_alloc(C.nbytes)
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)
        mod = SourceModule(kernel_code)
        kernel = mod.get_function("kernel")
        threads = (16, 16, 1)
        grid = (int((P + 15) // 16), int((M + 15) // 16))
        kernel(C_gpu, A_gpu, B_gpu, np.int32(M), np.int32(N), np.int32(P), block=threads, grid=grid)
        cuda.memcpy_dtoh(C, C_gpu)
        print("Result C (first 5x5 block):\n", C[:5, :5])
    else:
        # Vector operation example
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        out = np.empty_like(a)
        out_gpu = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)
        mod = SourceModule(kernel_code)
        kernel = mod.get_function("kernel")
        threads = 256
        blocks = int((n + threads - 1) // threads)
        kernel(out_gpu, a_gpu, b_gpu, np.int32(n), block=(threads,1,1), grid=(blocks,1))
        cuda.memcpy_dtoh(out, out_gpu)
        print("Result out (first 10 elements):", out[:10])

if __name__ == "__main__":
    run_dsl_file("vector_add.dsl")
