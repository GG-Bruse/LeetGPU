#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) 
{
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 in4 = reinterpret_cast<const float4*>(input)[idx];
        float4 out4;
        out4.x = fmaxf(in4.x, 0.0f);
        out4.y = fmaxf(in4.y, 0.0f);
        out4.z = fmaxf(in4.z, 0.0f);
        out4.w = fmaxf(in4.w, 0.0f);
        reinterpret_cast<float4*>(output)[idx] = out4;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            output[t] = fmaxf(input[t], 0.0f);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int N4 = N >> 2;
    int totalThread = N4 + (N % 4);
    int blocksPerGrid = (totalThread + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
