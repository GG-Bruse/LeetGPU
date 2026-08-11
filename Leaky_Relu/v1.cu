#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) 
{
    float alpha = 0.01f;
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 in4 = reinterpret_cast<const float4*>(input)[idx];
        float4 out4;
        out4.x = in4.x > 0 ? in4.x : in4.x * alpha;
        out4.y = in4.y > 0 ? in4.y : in4.y * alpha;
        out4.z = in4.z > 0 ? in4.z : in4.z * alpha;
        out4.w = in4.w > 0 ? in4.w : in4.w * alpha;
        reinterpret_cast<float4*>(output)[idx] = out4;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            output[t] = input[t] > 0 ? input[t] : alpha * input[t];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int N4 = N >> 2;
    int totalThreads = N4 + (N % 4);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
