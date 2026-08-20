#include <cuda_runtime.h>

__device__ __forceinline__ float silu(float x) {
    return __fdividef(x, 1.0f + __expf(-x));   // x / (1 + e^-x)，全快速路径
}

__global__ void silu_kernel(const float* input, float* output, int N) 
{
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 inTmp = reinterpret_cast<const float4*>(input)[idx];
        float4 outTmp;
        outTmp.x = silu(inTmp.x);
        outTmp.y = silu(inTmp.y);
        outTmp.z = silu(inTmp.z);
        outTmp.w = silu(inTmp.w);
        reinterpret_cast<float4*>(output)[idx] = outTmp;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            output[t] = silu(input[t]);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if(N <= 0) return;
    int threadsPerBlock = 256;
    int N4 = N >> 2;
    int totalThreads = N4 + (N % 4);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
