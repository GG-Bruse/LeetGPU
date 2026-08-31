#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define THREADS 256

__global__ void matrix_multiplication_kernel(const half* A, const half* B, half* output, int M, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    if(row < M && col < N) {
        float sum = 0.0f;
        for(int i = 0; i < K; ++i) {
            // A的 (row + 1)行 i 列 * B的 (i + 1)行 col 列
            sum += __half2float(A[row * K + i]) * __half2float(B[i * N + col]);
        }
        output[row * N + col] = sum;
    }
}

__global__ void multiplication_kernel(half* inout, float factor, int M, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    if(row < M && col < N) {
        inout[row * N + col] = __half2float(inout[row * N + col]) * factor;
    }
}

__global__ void matrix_and(half* A, half* B, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    if(row < M && col < N) {
        B[row * N + col] += A[row * N + col];
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) 
{
    if(M <= 0 || N <= 0 || K <= 0) return;
    half* tmp = nullptr;
    cudaMalloc(&tmp, sizeof(half) * (M * N));

    int totalThreads = M * N;
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    matrix_multiplication_kernel<<<blocksPerGrid, THREADS>>>(A, B, tmp, M, N, K);
    multiplication_kernel<<<blocksPerGrid, THREADS>>>(tmp, alpha, M, N);
    multiplication_kernel<<<blocksPerGrid, THREADS>>>(C, beta, M, N);
    matrix_and<<<blocksPerGrid, THREADS>>>(tmp, C, M, N);

    cudaFree(tmp);
    cudaDeviceSynchronize();
}
