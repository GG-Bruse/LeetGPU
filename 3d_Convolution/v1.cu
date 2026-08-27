#include <cuda_runtime.h>
#define THREADS 256

__global__ void convolution_3d_kernel(const float* input, const float* kernel, float* output, int input_depth,
    int input_rows, int input_cols, int kernel_depth, int kernel_rows,
    int kernel_cols, int output_depth, int output_rows, int output_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_rows * output_cols * output_depth) return;
    int depth = idx / (output_rows * output_cols);
    int row = idx % (output_rows * output_cols) / output_cols;
    int col = idx % (output_rows * output_cols) % output_cols;
    float sum = 0.0f;
    for(int i = 0; i < kernel_depth; ++i) {
        for(int j = 0; j < kernel_rows; ++j) {
            for(int k = 0; k < kernel_cols; ++k) {
                sum += (input[(i + depth) * (input_rows * input_cols) + (row + j) * input_cols + (col + k)] * kernel[i * (kernel_rows * kernel_cols) + j * kernel_cols + k]);
            }
        }
    }
    output[depth * (output_rows * output_cols) + row * output_cols + col] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) 
{
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    int totalThreads = output_rows * output_cols * output_depth;
    int blocksPerGrid = (totalThreads + THREADS - 1) / THREADS;
    convolution_3d_kernel<<<blocksPerGrid, THREADS>>>(input, kernel, output, \
        input_depth, input_rows, input_cols, \
        kernel_depth, kernel_rows, kernel_cols, \
        output_depth, output_rows, output_cols);
    cudaDeviceSynchronize();
}
