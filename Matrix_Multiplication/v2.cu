#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    //shared memory 是每个 block 私有的
    // grid 里 100 个 block, 就有 100 份独立的 As/Bs，各用各的，互不可见
    // __syncthreads() 也只管本 block 内部 —— block 之间根本不需要同步，因为它们连 shared memory 都不共享
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    //     N 维（滑动方向 →）
    // A: [tile0][tile1][tile2]   第 row 行
    //         ↓ 各与对应位置的 B tile 相乘
    // B: [tile0]
    //    [tile1]                 第 col 列
    //    [tile2]
    for(int t = 0; t < (N + TILE - 1) / TILE; ++t)
    {
        int aCol = t * TILE + threadIdx.x; // A 中当前 tile 的列
        int bRow = t * TILE + threadIdx.y; // B 中当前 tile 的行
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < K && bRow < N) ? B[bRow * K + col] : 0.0f;
        __syncthreads();

        for(int i = 0; i < TILE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // A -> M行,N列   B -> N行,K列   C -> M行,K列
    dim3 threadsPerBlock(16, 16);
    // K 列 \ M 行 (都向上取整)
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
