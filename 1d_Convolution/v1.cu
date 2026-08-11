#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) 
{
    // 动态 shared memory：缓存整个卷积核
    extern __shared__ float s_kernel[];
    for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
        s_kernel[j] = kernel[j];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if(idx < output_size) {
        float sum = 0.0f;
        for(int j = 0; j < kernel_size; ++j) {
            sum += input[idx + j] * s_kernel[j];
        }
        output[idx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    if (output_size <= 0) return;

    // 每个线程负责一个 output位置 
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedBytes = kernel_size * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
