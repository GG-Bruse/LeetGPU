#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < (N >> 1)) {
        int j = N - idx - 1;
        float tmp = input[idx];
        input[idx] = input[j];
        input[j] = tmp; 
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int half = N >> 1;
    int blocksPerGrid = (half + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
