#include <cuda_runtime.h>
#define THREADS 256

__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    atomicAdd(&result[0], A[idx] * B[idx]);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) 
{
    if(N <= 0) return;
    int blocksPerGrid = (N + THREADS - 1) / THREADS;
    dot_product_kernel<<<blocksPerGrid, THREADS>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
