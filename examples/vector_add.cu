__global__ void kernel(float* out, float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        out[i] = (a[i] + (b[i] * 2));
    }
}
