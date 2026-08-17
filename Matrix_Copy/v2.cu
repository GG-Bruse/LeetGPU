#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N) {
        B[row * N + col] = A[row * N + col];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
