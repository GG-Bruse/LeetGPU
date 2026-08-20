#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) 
{
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 inTmp = reinterpret_cast<const float4*>(input)[idx];
        float4 outTmp;
        outTmp.x = fminf(fmaxf(inTmp.x, lo), hi);
        outTmp.y = fminf(fmaxf(inTmp.y, lo), hi);
        outTmp.z = fminf(fmaxf(inTmp.z, lo), hi);
        outTmp.w = fminf(fmaxf(inTmp.w, lo), hi);
        reinterpret_cast<float4*>(output)[idx] = outTmp;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            output[t] = fminf(fmaxf(input[t], lo), hi);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    if(N <= 0) return;
    int N4 = N >> 2;
    int threadsPerBlock = 256;
    int totalThreads = N4 + (N % 4);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
