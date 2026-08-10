#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) 
{
    int total = N * N;
    int N4F = total >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4F) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[idx] = c;
    }
    else {
        int t = (N4F << 2) + (idx - N4F);
        if(t < total) {
            C[t] = A[t] + B[t];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int total = N * N;
    int N4F = total >> 2;
    int totalThreads = N4F + (total & 3);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
