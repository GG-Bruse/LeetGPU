#include <cuda_runtime.h>
#define THREADS 256

__global__ void histogramming_kernel(const int* input, int* histogram, int N, int num_bins) 
{
    extern __shared__ int s_hist[]; // 块内私有直方图

    // 块内块内私有直方图置0
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        s_hist[b] = 0;
    __syncthreads();

    // 块内统计：atomicAdd 打在 shared memory 上, grid-stride 统计：每个线程跨步处理多个 int4
    int N4 = N >> 2;
    int stride = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N4; i += stride) {
        int4 v = reinterpret_cast<const int4*>(input)[i];
        atomicAdd(&s_hist[v.x], 1);
        atomicAdd(&s_hist[v.y], 1);
        atomicAdd(&s_hist[v.z], 1);
        atomicAdd(&s_hist[v.w], 1);
    }
    if (idx < (N % 4)) // 尾巴：只放前 (N % 4) 个线程进来，idx 本身就是尾巴内偏移，所以不用减 N4
        atomicAdd(&s_hist[input[(N4 << 2) + idx]], 1);
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

    // SM 数只查一次，static 缓存住（solve 会被多次调用）
    static int sms = 0;
    if (sms == 0)
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);

    int N4 = N >> 2;
    int totalThreads = N4 + (N % 4);
    int needed = (totalThreads + THREADS - 1) / THREADS;
    int blocks = sms * 4 < needed ? sms * 4 : needed;

    size_t sharedBytes = num_bins * sizeof(int);
    histogramming_kernel<<<blocks, THREADS, sharedBytes>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
