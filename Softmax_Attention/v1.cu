#include <cuda_runtime.h>
#define THREADS 256

struct MS { float max; float sum; };

__device__ __forceinline__ MS merge(MS a, MS b) {
    if (a.max == -INFINITY) return b;
    if (b.max == -INFINITY) return a;
    float m = fmaxf(a.max, b.max);
    float s = a.sum * __expf(a.max - m) + b.sum * __expf(b.max - m);
    return { m, s };
}

// kernel 1：S[i][j] = (Q_i · K_j) / √d
// Q M * d 
// K N * d -> K^T d * N
// Q * K^T = M * N
__global__ void scores_kernel(const float* Q, const float* K, float* S, float scale, int M, int N, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= M * N) return;
    int row = idx / N; // S 的 行
    int col = idx % N; // S 的 列
    const float* q = Q + (size_t)row * d;
    const float* k = K + (size_t)col * d;
    float s = 0.0f;
    for (int t = 0; t < d; ++t) s += q[t] * k[t];
    S[row * N + col] = s * scale;
}

// ---- kernel 2：一个 block 负责一行 S，online 规约出该行的 (max, sum) ----
__global__ void row_stats_kernel(const float* S, MS* rowStats, int M, int N) 
{
    __shared__ MS sdata[THREADS];
    int row = blockIdx.x;
    if (row >= M) return;

    const float* s = S + (size_t)row * N;
    int tid = threadIdx.x;

    MS local { -INFINITY, 0.0f};
    for(int i = tid; i < N; i += THREADS) { // 行内跨步扫描, 每行N个元素都归到一个block中
        float x = s[i];
        float max_new = fmaxf(local.max, x);
        local.sum = local.sum * __expf(local.max - max_new) + __expf(x - max_new);
        local.max = max_new;
    }
    sdata[tid] = local;
    __syncthreads();

    for(int offset = THREADS / 2; offset > 0; offset >>= 1) {
        if(tid < offset) sdata[tid] = merge(sdata[tid], sdata[tid + offset]);
        __syncthreads();
    }
    if(tid == 0) rowStats[row] = sdata[tid];
}

// ---- kernel 3：output[i][t] = Σ_j softmax(S)_ij · V[j][t]，一线程一个输出 
// softmax(S) M * N
// V N × d
// softmax(S) * V -> M * d
__global__ void apply_kernel(const float* S, const MS* rowStats, const float* V, float* output, int M, int N, int d) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * d) return;

    int row = idx / d; // output 行
    int col = idx % d; // output 列
    MS local = rowStats[row];
    float max = local.max;
    float inv_sum = 1.0f / local.sum;
    const float* s = S + (size_t)row * N;

    float acc = 0.0f;
    for(int j = 0; j < N; ++j) {
        acc += __expf(s[j] - max) * V[j * d + col];
    }
    output[idx] = acc * inv_sum;
}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) 
{
    if (M <= 0 || N <= 0 || d <= 0) return;
    float* score = nullptr; // M × N 分数矩阵
    MS* stats = nullptr; // 每行的 (max, sum)
    cudaMalloc(&score, (size_t)M * N * sizeof(float));
    cudaMalloc(&stats, M * sizeof(MS));

    int scoreThreads = M * N;
    float scale = 1.0f / sqrtf((float)d);
    int scoreBlocks = (scoreThreads + THREADS - 1) / THREADS;
    scores_kernel<<<scoreBlocks, THREADS>>>(Q, K, score, scale, M, N, d);

    row_stats_kernel<<<M, THREADS>>>(score, stats, M, N);

    int outThreads = M * d;
    int outBlocks = (outThreads + THREADS - 1) / THREADS;
    apply_kernel<<<outBlocks, THREADS>>>(score, stats, V, output, M, N, d);

    cudaFree(score);
    cudaFree(stats);
    cudaDeviceSynchronize();
}