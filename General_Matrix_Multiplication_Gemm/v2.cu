#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define THREADS 256

__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int row = idx / N;
    int col = idx % N;
    float sum = 0.0f;
    // A的 (row + 1)行 i 列 * B的 (i + 1)行 col 列
    for(int i = 0; i < K; ++i) 
        sum += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
    C[idx] = __float2half(alpha * sum + beta * __half2float(C[idx]));
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) 
{
    if(M <= 0 || N <= 0 || K <= 0) return;
    int totalThreads = M * N;
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    gemm_kernel<<<blocksPerGrid, THREADS>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
