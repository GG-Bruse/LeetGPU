#include <cuda_runtime.h>
#define THREADS 256

struct MS { float max; float sum; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    if (a.max == -INFINITY) return b;   // 空段防护
    if (b.max == -INFINITY) return a;
    float m = fmaxf(a.max, b.max);
    float s = a.sum * __expf(a.max - m) + b.sum * __expf(b.max - m);
    return { m, s };
}

// 一趟扫描同时出局部 max 和局部指数和
__global__ void block_stats_kernel(const float* input, MS* part, int N) 
{
    __shared__ MS sdata[THREADS];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    MS local = { -INFINITY, 0.0f };
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float x = input[i];
        float m_new = fmaxf(local.max, x);
        local.sum = local.sum * __expf(local.max - m_new) + __expf(x - m_new);
        local.max = m_new;
    }
    sdata[tid] = local;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] = merge(sdata[tid], sdata[tid + offset]);
        __syncthreads();
    }

    if(tid == 0) part[blockIdx.x] = sdata[0];
}

// 单 block 把 (m, s) 对规约成全局值
__global__ void final_stats_kernel(MS* part, float* scalars, int blocks)
{
    __shared__ MS sdata[THREADS];
    int tid = threadIdx.x;

    MS local = { -INFINITY, 0.0f };
    for(int i = tid; i < blocks; i += THREADS) {
        local = merge(local, part[i]);
    }
    sdata[tid] = local;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] = merge(sdata[tid], sdata[tid + offset]);
        __syncthreads();
    }
    if(tid == 0) {
        scalars[0] = sdata[0].max;
        scalars[1] = sdata[0].sum;
    }
}

__global__ void softmax_kernel(const float* input, float* output, float* scalars, int N)
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

    MS* part = nullptr;
    float* scalars = nullptr;
    cudaMalloc(&part, blocksPerGrid * sizeof(MS));
    cudaMalloc(&scalars, 2 * sizeof(float));

    block_stats_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, part, N);
    final_stats_kernel<<<1, threadsPerBlock>>>(part, scalars, blocksPerGrid);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, scalars, N);

    cudaFree(part);
    cudaFree(scalars);
    cudaDeviceSynchronize();
}
