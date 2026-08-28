#include <cuda_runtime.h>
#define THREADS 256

__global__ void sparse_matrix_vector_mul_kernel(const float* A, const float* x, float* y, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < M) {
        float sum = 0.0f;
        for(int i = 0; i < N; ++i) {
            sum += (A[idx * N + i] * x[i]);
        }
        y[idx] = sum;
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) 
{
    if(M <= 0 || N <= 0) return;
    int totalThreads = M;
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    sparse_matrix_vector_mul_kernel<<<blocksPerGrid, THREADS>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
