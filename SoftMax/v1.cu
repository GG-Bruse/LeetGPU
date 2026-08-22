#include <cuda_runtime.h>
#define THREADS 256

// 每 block 跨步扫一段，块内规约出局部 max
__global__ void block_max_kernel(const float* input, float* part, int N) 
{
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = -INFINITY;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) max_val = fmaxf(max_val, input[i]);
    sdata[tid] = max_val;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        __syncthreads();
    }

    if(tid == 0) part[blockIdx.x] = sdata[tid];
}

// 单 block 把 partial 规约成全局 max
__global__ void final_max_kernel(float* part, float* scalars, int blocks)
{
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;

    float max_val = -INFINITY;
    for(int i = tid; i < blocks; i += THREADS) max_val = fmaxf(max_val, part[i]);
    sdata[tid] = max_val;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        __syncthreads();
    }

    if(tid == 0) scalars[0] = sdata[tid];
}

// 读全局 max，每 block 求局部指数和
__global__ void block_sum_kernel(const float* input, float* part, float* scalars, int N) 
{
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = scalars[0];

    float sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) sum += __expf(input[i] - max_val);
    sdata[tid] = sum;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    if(tid == 0) part[blockIdx.x] = sdata[0];
}

// 单 block 把 partial 规约成全局 sum
__global__ void final_sum_kernel(float* part, float* scalars, int blocks)
{
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for(int i = tid; i < blocks; i += THREADS) sum += part[i];
    sdata[tid] = sum;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }
    if(tid == 0) scalars[1] = sdata[tid];
}

__global__ void softmax_kernel(const float* input, float* scalars, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = scalars[0];
    float inv_sum = 1.0f / scalars[1]; 
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = __expf(input[i] - max_val) * inv_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    if(N <= 0) return;
    int threadsPerBlock = THREADS;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if(blocksPerGrid > 1024) blocksPerGrid = 1024;

    float* part = nullptr;
    float* scalars = nullptr;   // [0] = 全局 max, [1] = 全局 指数sum
    cudaMalloc(&part, blocksPerGrid * sizeof(float));
    cudaMalloc(&scalars, 2 * sizeof(float));

    block_max_kernel<<<blocksPerGrid, THREADS>>>(input, part, N);
    final_max_kernel<<<1, THREADS>>>(part, scalars, blocksPerGrid);
    block_sum_kernel<<<blocksPerGrid, THREADS>>>(input, part, scalars, N);
    final_sum_kernel<<<1, THREADS>>>(part, scalars, blocksPerGrid);
    softmax_kernel<<<blocksPerGrid, THREADS>>>(input, scalars, output, N);

    cudaFree(part);
    cudaFree(scalars);
    cudaDeviceSynchronize();
}
