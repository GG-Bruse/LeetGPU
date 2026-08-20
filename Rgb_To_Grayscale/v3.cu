#include <cuda_runtime.h>

__device__ __forceinline__ float rgb2gray(float r, float g, float b) {
    return r * 0.299f + g * 0.587f + b * 0.114f;
}

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) 
{
    int total_pixels = width * height;
    int total_pixels_N4 = total_pixels >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_pixels_N4) {
        const float4* inTmp = reinterpret_cast<const float4*>(input);
        float4 v1 = inTmp[3 * idx];
        float4 v2 = inTmp[3 * idx + 1];
        float4 v3 = inTmp[3 * idx + 2];
        float4 out;
        out.x = rgb2gray(v1.x, v1.y, v1.z);
        out.y = rgb2gray(v1.w, v2.x, v2.y);
        out.z = rgb2gray(v2.z, v2.w, v3.x);
        out.w = rgb2gray(v3.y, v3.z, v3.w);
        reinterpret_cast<float4*>(output)[idx] = out;
    } else {
        int t = (total_pixels_N4 << 2) + (idx - total_pixels_N4);
        if(t < total_pixels) {
            output[t] = rgb2gray(input[3 * t], input[3 * t + 1], input[3 * t + 2]);
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int total_pixels_N4 = total_pixels >> 2;
    int threadsPerBlock = 256;
    int totalThreads = total_pixels_N4 + (total_pixels % 4);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
