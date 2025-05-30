import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from frontend.parser import parse_input
from codegen.cuda_codegen import generate_kernel

def run_vector_kernel(expr, a, b=None):
    ast = parse_input(expr)
    kernel_code = generate_kernel(ast)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    if b is not None:
        b_gpu = cuda.mem_alloc(b.nbytes)
        cuda.memcpy_htod(b_gpu, b)
    out = np.empty_like(a)
    out_gpu = cuda.mem_alloc(out.nbytes)
    mod = SourceModule(kernel_code)
    kernel = mod.get_function("kernel")
    n = a.size
    threads = 256
    blocks = int((n + threads - 1) // threads)  # Ensure Python int
    if b is not None:
        kernel(out_gpu, a_gpu, b_gpu, np.int32(n), block=(threads,1,1), grid=(blocks,1))
    else:
        kernel(out_gpu, a_gpu, np.int32(n), block=(threads,1,1), grid=(blocks,1))
    cuda.memcpy_dtoh(out, out_gpu)
    return out

def run_matmul_kernel(expr, A, B):
    ast = parse_input(expr)
    kernel_code = generate_kernel(ast)
    M, N = A.shape
    N2, P = B.shape
    assert N == N2
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C = np.empty((M, P), dtype=np.float32)
    C_gpu = cuda.mem_alloc(C.nbytes)
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)
    mod = SourceModule(kernel_code)
    kernel = mod.get_function("kernel")
    threads = (16, 16, 1)
    grid = (int((P + 15) // 16), int((M + 15) // 16))  # Ensure Python ints
    kernel(C_gpu, A_gpu, B_gpu, np.int32(M), np.int32(N), np.int32(P), block=threads, grid=grid)
    cuda.memcpy_dtoh(C, C_gpu)
    return C

def test_all():
    # 1. Addition
    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)
    out = run_vector_kernel("out = a + b", a, b)
    print("Addition error:", np.max(np.abs(out - (a + b))))

    # 2. Subtraction
    out = run_vector_kernel("out = a - b", a, b)
    print("Subtraction error:", np.max(np.abs(out - (a - b))))

    # 3. Multiplication
    out = run_vector_kernel("out = a * b", a, b)
    print("Multiplication error:", np.max(np.abs(out - (a * b))))

    # 4. Division
    b2 = b + 1.0  # avoid zero division
    out = run_vector_kernel("out = a / b", a, b2)
    print("Division error:", np.max(np.abs(out - (a / b2))))

    # 5. Matrix multiplication
    A = np.random.randn(64, 128).astype(np.float32)
    B = np.random.randn(128, 32).astype(np.float32)
    C = run_matmul_kernel("C = A @ B", A, B)
    print("Matmul error:", np.max(np.abs(C - np.dot(A, B))))

if __name__ == "__main__":
    test_all()
