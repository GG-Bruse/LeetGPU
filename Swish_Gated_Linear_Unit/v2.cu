#include <cuda_runtime.h>

__device__ __forceinline__ float swiglu(float x1, float x2) {
    return __fdividef(x1, 1 + __expf(-x1)) * x2;
}

__global__ void swiglu_kernel_vec4(const float* input, float* output, int halfN4) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN4) {
        float4 a = reinterpret_cast<const float4*>(input)[idx];
        float4 b = reinterpret_cast<const float4*>(input + (halfN4 << 2))[idx];
        float4 outTmp;
        outTmp.x = swiglu(a.x, b.x);
        outTmp.y = swiglu(a.y, b.y);
        outTmp.z = swiglu(a.z, b.z);
        outTmp.w = swiglu(a.w, b.w);
        reinterpret_cast<float4*>(output)[idx] = outTmp;
    }
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < halfN) {
        output[idx] = swiglu(input[idx], input[idx + halfN]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if(N <= 1 || N % 2 != 0) return;
    int halfN = N / 2;
    int threadsPerBlock = 256;
    if(halfN % 4 == 0) {
        int halfN4 = halfN >> 2;
        int blocksPerGrid = (halfN4 + threadsPerBlock - 1) / threadsPerBlock;
        swiglu_kernel_vec4<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN4);
    } else {
        int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
        swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    }
    cudaDeviceSynchronize();
}
