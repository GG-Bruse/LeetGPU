#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) 
{
    // +1 列 padding：消除 shared memory 的 bank conflict
    __shared__ float tile[TILE][TILE + 1];

    // 按 input 的布局，合并读入 tile
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if(row < rows && col < cols) 
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
    __syncthreads();

    // 按 output 的布局，从 tile 转置读出、合并写出
    // 转置后 block 的行列角色互换：output 的块坐标 = (blockIdx.x, blockIdx.y) 对调
    int outRow = blockIdx.x * TILE + threadIdx.y;
    int outCol = blockIdx.y * TILE + threadIdx.x;
    if (outRow < cols && outCol < rows)
        output[outRow * rows + outCol] = tile[threadIdx.x][threadIdx.y];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
