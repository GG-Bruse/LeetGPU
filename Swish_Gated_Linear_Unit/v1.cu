#include <cuda_runtime.h>

__device__ __forceinline__ float swiglu(float x1, float x2) {
    return __fdividef(x1, 1 + __expf(-x1)) * x2;
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
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
