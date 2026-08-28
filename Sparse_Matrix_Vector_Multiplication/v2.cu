#include <cuda_runtime.h>
#define THREADS 256

__global__ void sparse_matrix_vector_mul_kernel(const float* A, const float* x, float* y, int M, int N)
{
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < N; i += blockDim.x) // 存在重复读取
        sx[i] = x[i];
    __syncthreads();

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warpId >= M) return;
    int lane = threadIdx.x % 32;

    const float* row = A + (size_t)warpId * N;
    float sum = 0.0f;
    for (int j = lane; j < N; j += 32) 
        sum += row[j] * sx[j];   

    for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) y[warpId] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) 
{
    if(M <= 0 || N <= 0) return;
    int warpsPerBlock = THREADS / 32; // 8
    int blocksPerGrid = (M + warpsPerBlock - 1) / warpsPerBlock; // 一行一个warp
    sparse_matrix_vector_mul_kernel<<<blocksPerGrid, THREADS, sizeof(float) * N>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
