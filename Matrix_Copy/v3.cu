#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total) 
{
    int total4 = total >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total4) {
        reinterpret_cast<float4*>(B)[idx] = reinterpret_cast<const float4*>(A)[idx];
    } else {
        int t = (total4 << 2) + (idx - total4);
        if(t < total) {
            B[t] = A[t];
        }
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    if(total <= 0) return;

    int total4 = total >> 2;
    int threadsPerBlock = 256;
    int totalThreads = total4 + (total % 4);
    int blocksPerGrid = (totalThreads + threadsPerBlock -1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
