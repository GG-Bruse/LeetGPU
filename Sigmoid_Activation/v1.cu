#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float sigmoid(float x) {
    return __fdividef(1.0f, 1.0f + __expf(-x));
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) 
{
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 xTmp = reinterpret_cast<const float4*>(X)[idx];
        float4 yTmp;
        yTmp.x = sigmoid(xTmp.x);
        yTmp.y = sigmoid(xTmp.y);
        yTmp.z = sigmoid(xTmp.z);
        yTmp.w = sigmoid(xTmp.w);
        reinterpret_cast<float4*>(Y)[idx] = yTmp;
    } else {
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            Y[t] = sigmoid(X[t]);
        }
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    if(N <= 0) return;
    int N4 = N >> 2;
    int threadsPerBlock = 256;
    int totalThreads = N4 + (N % 4);
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
