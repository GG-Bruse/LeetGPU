#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) 
{
    int total_pixels = width * height;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_pixels) {
        float3 inTmp = reinterpret_cast<const float3*>(input)[idx];
        float outTmp;
        outTmp = inTmp.x * 0.299f + inTmp.y * 0.587f + inTmp.z * 0.114f;
        output[idx] = outTmp;
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
