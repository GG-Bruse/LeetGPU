#include <cuda_runtime.h>
#define THREADS 256

__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N)
{
    __shared__ float sdata[THREADS]; 
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int N4 = N >> 2;
    float prod = 0.0f;
    if (idx < N4) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        prod = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) prod = A[t] * B[t];
    }
    sdata[tid] = prod;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }

    if(tid == 0) atomicAdd(&result[0], sdata[0]);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) 
{
    if(N <= 0) return;
    int N4 = N >> 2;
    int totalThreads = N4 + (N % 4);
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    dot_product_kernel<<<blocksPerGrid, THREADS>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
