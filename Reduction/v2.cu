#include <cuda_runtime.h>

__global__ void Reduction_Kernel(const float* input, float* output, int N)
{
    __shared__ float sdata[8];   // 每个 warp 一个槽，256 线程 = 8 个 warp

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程跨步累加多个元素, 让所有 block 平分整个数组
    float sum = 0.0f;
    int strideAll = blockDim.x * gridDim.x;
    for(int i = idx; i < N; i += strideAll) {
        sum += input[i];
    }

    // warp 内用 shuffle 规约：5 步，零 shared memory、零同步
    // warp 内部（shuffle 的作用范围）
    // 32 个 lane 执行同一条 shuffle 指令时，硬件保证 mask 里点名的所有 lane 都到达这条指令后才执行。也就是说，当 shuffle 发生时，每个 lane 的 sum 必然是它循环全部跑完后的终值——不可能出现「lane 5 还在累加，lane 0 就来取它的值」
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    // 每个 warp 的 lane0 将 warp和 写入 sdata
    int lane = tid & 31; // warp内编号
    int warp = tid >> 5; // 所在warp编号
    if(lane == 0) sdata[warp] = sum;
    __syncthreads();

    // warp 0 再把 8 个 warp和 规约成 block 总和
    if(warp == 0) {
        sum = tid < (blockDim.x >> 5) ? sdata[tid] : 0.0f;
        for(int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if(tid == 0) atomicAdd(output, sum);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    cudaMemset(output, 0, sizeof(float));
    int threadsPerBlock = 256;
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blockPerGrid > 1024) blockPerGrid = 1024;
    Reduction_Kernel<<<blockPerGrid, threadsPerBlock>>>(input, output, N);
    return;
}
