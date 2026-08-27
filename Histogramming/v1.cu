#include <cuda_runtime.h>
#define THREADS 256

__global__ void histogramming_kernel(const int* input, int* histogram, int N, int num_bins) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    atomicAdd(&histogram[input[idx]], 1);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) 
{
    if(N <= 0) return;
    int blocksPerGrid = (N + THREADS - 1) / THREADS;
    histogramming_kernel<<<blocksPerGrid, THREADS>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
