#include <cuda_runtime.h>
#define THREADS 256

__global__ void histogramming_kernel(const int* input, int* histogram, int N, int num_bins) 
{
    extern __shared__ int s_hist[]; // 块内私有直方图

    // 块内块内私有直方图置0
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        s_hist[b] = 0;
    __syncthreads();

    // 块内统计：atomicAdd 打在 shared memory 上
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        int4 v = reinterpret_cast<const int4*>(input)[idx];
        atomicAdd(&s_hist[v.x], 1);
        atomicAdd(&s_hist[v.y], 1);
        atomicAdd(&s_hist[v.z], 1);
        atomicAdd(&s_hist[v.w], 1);
    } else {
        int t = (N4 << 2) + (idx - N4);
        if (t < N) atomicAdd(&s_hist[input[t]], 1);
    }
    __syncthreads();
    // 把私有直方图累加进全局
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        if (s_hist[b] > 0)                    // 空 bin 跳过，白省一次全局原子操作
            atomicAdd(&histogram[b], s_hist[b]);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) 
{
    cudaMemset(histogram, 0, num_bins * sizeof(int));
    if(N <= 0) return;
    int N4 = N >> 2;
    int totalThreads = N4 + (N % 4);
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    size_t sharedBytes = num_bins * sizeof(int);
    histogramming_kernel<<<blocksPerGrid, THREADS, sharedBytes>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
