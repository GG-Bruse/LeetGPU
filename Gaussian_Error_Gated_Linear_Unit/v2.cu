#include <cuda_runtime.h>

__device__ float geglu(float a, float b) 
{
    return a * (b * 0.5f * (1.0f + erff(b * 0.70710678118654752f)));
}

__global__ void geglu_kernel_vec4(const float* input, float* output, int halfN) 
{
    int halfN4 = halfN >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN4) {
        float4 a = reinterpret_cast<const float4*>(input)[idx];
        float4 b = reinterpret_cast<const float4*>(input + halfN)[idx];
        float4 out;
        out.x = geglu(a.x, b.x);
        out.y = geglu(a.y, b.y);
        out.z = geglu(a.z, b.z);
        out.w = geglu(a.w, b.w);
        reinterpret_cast<float4*>(output)[idx] = out;
    }
}

__global__ void geglu_kernel(const float* input, float* output, int halfN) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
        output[idx] = geglu(input[idx], input[idx + halfN]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if(N <= 0 || N % 2 != 0) return;
    int halfN = N / 2;
    int halfN4 = halfN >> 2;
    int threadsPerBlock = 256;

    if(halfN % 4 == 0) {
        int blocksPerGrid = (halfN4 + threadsPerBlock - 1) / threadsPerBlock;
        geglu_kernel_vec4<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    } else {
        int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
        geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    }
    
    cudaDeviceSynchronize();
}
