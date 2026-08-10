#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) 
{
    int total = width * height;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < total) 
        reinterpret_cast<unsigned int*>(image)[index] ^= 0x00FFFFFFu;
        // 一个像素 = 4 字节 = 一个 32 位字
        // ^ 0x00FFFFFF：RGB 三字节取反（等价 255 - v），Alpha 字节不变
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
