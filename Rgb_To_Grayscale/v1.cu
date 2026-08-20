#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) 
{
    int total_pixels = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_pixels) {
        output[idx] = input[3 * idx] * 0.299f + input[3 * idx + 1] * 0.587 + input[3 * idx + 2] * 0.114;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
