#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) 
{
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 lo, hi;
        lo.x = a.x;  lo.y = b.x;  lo.z = a.y;  lo.w = b.y;   // A0 B0 A1 B1
        hi.x = a.z;  hi.y = b.z;  hi.z = a.w;  hi.w = b.w;   // A2 B2 A3 B3
        float4* out4 = reinterpret_cast<float4*>(output);
        out4[2 * idx]     = lo;   // 128 位连续写
        out4[2 * idx + 1] = hi;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            output[t * 2] = A[t];
            output[t * 2 + 1] = B[t];
        } 
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    if(N <= 0) return;
    int N4 = N >> 2;
    int totalThreads = N4 + (N % 4);
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
