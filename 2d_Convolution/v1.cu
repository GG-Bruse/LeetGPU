#include <cuda_runtime.h>
#define THREADS 256

__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output, int input_rows,
    int input_cols, int kernel_rows, int kernel_cols, int output_rows, int output_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_rows * output_cols) return;
    int row = idx / output_cols;
    int col = idx % output_cols;
    float sum = 0.0f;
    for(int i = 0; i < kernel_rows; ++i) {
        for(int j = 0; j < kernel_cols; ++j) {
            sum += (input[(row + i) * input_cols + (col + j)] * kernel[i * kernel_cols + j]);
        }
    }
    output[row * output_cols + col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) 
{
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    int totalThreads = output_rows * output_cols;
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    convolution_2d_kernel<<<blocksPerGrid, THREADS>>>(input, kernel, output, input_rows, input_cols,
        kernel_rows, kernel_cols, output_rows, output_cols);
    cudaDeviceSynchronize();
}
