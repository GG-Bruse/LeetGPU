#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int N4 = N >> 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N4) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[idx] = c;
    } else {
        // 计算尾部元素在原数组中的真实下标
        // idx > N4 && idx <= N4 + (N & 3) 
        // 0 < (idx - N4) <= (N & 3)
        int t = (N4 << 2) + (idx - N4);
        if(t < N) {
            C[t] = A[t] + B[t];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    // float4的总个数
    int N4 = N >> 2;
    // N & 3 等价于 N % 4
    int totalThreads = N4 + (N & 3);
    // 若是 N = 10, 即 8 个元素走向量化路径, 剩余两个

    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
