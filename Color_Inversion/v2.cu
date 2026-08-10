#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) 
{
    int numPixels = width * height;
    int N4 = numPixels >> 2;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N4) {
        uint4 p = reinterpret_cast<uint4*>(image)[index];
        p.x ^= 0x00FFFFFFu;
        p.y ^= 0x00FFFFFFu;
        p.z ^= 0x00FFFFFFu;
        p.w ^= 0x00FFFFFFu;
        reinterpret_cast<uint4*>(image)[index] = p;
    } else {
        int t = (N4 << 2) + (index - N4);
        if(t < numPixels) {
            reinterpret_cast<unsigned int*>(image)[t] ^= 0x00FFFFFFu;
        }
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int numPixels = width * height;// 像素数
    if (numPixels <= 0) return;

    int threadsPerBlock = 256;
    int N4 = numPixels >> 2;
    int totalThreads = N4 + (numPixels & 3);

    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
