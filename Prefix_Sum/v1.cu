#include <cuda_runtime.h>
#define THREADS 256

// 块内 inclusive scan（Hillis-Steele），并记录块总和
__global__ void local_scan_kernel(const float* input, float* output, float* blockSums, int N)
{
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = idx < N ? input[idx] : 0.0f;
    __syncthreads();

    for(int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = (tid >= offset) ? sdata[tid - offset] : 0.0f; // 先读进寄存器
        __syncthreads(); // 等待所有线程读取完毕
        sdata[tid] += val;
        __syncthreads();
    }

    if (idx < N) output[idx] = sdata[tid];
    if (tid == THREADS - 1) blockSums[blockIdx.x] = sdata[tid];
}

// local_scan_kernel 之后，blockSums[b] = 第 b 块的 256 个元素之和（块总和）；
// 本 kernel：把"块总和"数组原地变成"块偏移"数组 —— blockSums[b] 改写为第 0 ~ b-1 块的总和（不含自己）
__global__ void scan_block_sums_kernel(float* blockSums, int numBlocks) 
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float carry = 0.0f;

    for (int base = 0; base < numBlocks; base += 1024) // 每一轮处理1024个块
    {
        int i = base + tid;
        float x = (i < numBlocks) ? blockSums[i] : 0.0f;
        sdata[tid] = x;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            float val = (tid >= offset) ? sdata[tid - offset] : 0.0f;
            __syncthreads();
            sdata[tid] += val;
            __syncthreads();
        }

        if (i < numBlocks) blockSums[i] = carry + sdata[tid] - x;
        carry += sdata[1023];
        __syncthreads(); 
    }
}

// 每个块把这个偏移加到自己的每个元素上
__global__ void add_offsets_kernel(float* output, const float* blockSums, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] += blockSums[blockIdx.x];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) 
{
    if(N <= 0) return;
    int blocksPerGrid = (N + THREADS - 1) / THREADS;
    float* blockSums;
    cudaMalloc(&blockSums, blocksPerGrid * sizeof(float));

    local_scan_kernel<<<blocksPerGrid, THREADS>>>(input, output, blockSums, N);
    scan_block_sums_kernel<<<1, 1024>>>(blockSums, blocksPerGrid);
    add_offsets_kernel<<<blocksPerGrid, THREADS>>>(output, blockSums, N);

    cudaDeviceSynchronize();
}
