#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        output[2 * idx] = A[idx];
        output[2 * idx + 1] = B[idx];
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
