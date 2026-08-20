#include <cuda_runtime.h>

__device__ float geglu(float a, float b) 
{
    return a * (b / 2 * (1 + erf(b / sqrt(2))));
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
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
