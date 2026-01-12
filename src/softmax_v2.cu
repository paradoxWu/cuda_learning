#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <cmath>
const float device_infinity = HUGE_VALF;

std::vector<float> random_vector(std::size_t len, float min, float max)
{
    static std::random_device rd;                         // 非静态可重新播种
    static std::mt19937 gen(rd());                        // Mersenne Twister
    std::uniform_real_distribution<float> dist(min, max); // 任意范围
    std::vector<float> v(len);
    for (auto &x : v)
        x = dist(gen);
    return v;
}

__device__ float device_max(float a, float b) { return (a > b) ? a : b; }

inline __device__ float warpReduceMax(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val = device_max(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val; // 32 线程都是 max
}

inline __device__ float warpReduceSum(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val; // 32 线程都是 sum
}

__device__ float2 warpReduceMaxSum(float2 val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val.x = device_max(val.x, __shfl_xor_sync(0xffffffff, val.x, mask));
        val.y += __shfl_xor_sync(0xffffffff, val.y, mask);
    }
    return val;
}

__device__ float2 blockReduceMaxSum(float2 val)
{
    int tid = threadIdx.x;
    int warpsPerBlock = blockDim.x / 32; // 256 threads → 8 warps

    // 1. warp 内归约：32→1
    val = warpReduceMaxSum(val);

    // 2. 每 warp 首线程写回共享内存（仅 8 个 float2）
    extern __shared__ float2 shm[];
    int warpId = tid / 32;
    if (tid % 32 == 0)
        shm[warpId] = val;
    __syncthreads();

    // 3. 最后一个 warp 再把 8→1（0-shared 复用）
    if (warpId == 0)
    {
        val = (tid < warpsPerBlock) ? shm[tid] : make_float2(-device_infinity, 0.0f);
        val = warpReduceMaxSum(val);
        if (tid == 0)
            shm[0] = val; // 写回 shm[0]
    }
    __syncthreads();

    return shm[0]; // (global_max, global_sum)
}

// ========== 第一级：block 内 Online-Softmax ==========
__global__ void softmax_block(const float *g_in, float *g_exp, float *g_partial, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // 1. 在线 max（warpReduceMax）
    float val = (idx < n) ? g_in[idx] : -device_infinity;
    float maxVal = warpReduceMax(val);

    // 2. 在线 exp
    float expVal = (idx < n) ? __expf(val - maxVal) : 0.0f;

    // 3. 在线 sum（warpReduceSum）
    float sumVal = warpReduceSum(expVal);

    // 4. 写回：exp 值 + (max,sum) 给第二级
    if (idx < n)
        g_exp[idx] = expVal; // 逐元素 exp(x-max)
    if (tid == 0)
    {
        g_partial[blockIdx.x * 2 + 0] = maxVal;
        g_partial[blockIdx.x * 2 + 1] = sumVal;
    }
}

// ========== 第二级：grid 内归约 (max,sum) ==========
__global__ void softmax_final(const float *g_exp, const float *g_partial, float *g_out, int blocks)
{
    extern __shared__ float smax[]; // 2*blocks
    int tid = threadIdx.x;
    // 1. 读回所有 (max,sum)
    float2 local;
    local.x = -device_infinity;
    local.y = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x)
    {
        local.x = device_max(local.x, g_partial[i * 2 + 0]); // max
        local.y += g_partial[i * 2 + 1];                     // sum
    }
    // 2. 在线归约 (max,sum)
    float2 global = blockReduceMaxSum(local); // 你已有的 0-shared 实现
    // 3. 写回全局 (max,sum)
    if (tid == 0)
    {
        smax[0] = global.x;
        smax[1] = global.y;
    }
    __syncthreads();

    // 4. 逐元素除：exp(x-max) ÷ global_sum
    for (int i = tid; i < blocks * blockDim.x; i += blockDim.x)
    {
        float expVal = g_exp[i];
        g_out[i] = expVal / smax[1]; // 除以全局和
    }
}

void softmax_func(const float *in, float *out, int n)
{

    cudaFree(0);
    size_t bytes = n * sizeof(float);

    float *dev_in, *dev_out;
    cudaMalloc(&dev_in, bytes);
    cudaMalloc(&dev_out, bytes);
    cudaMemcpy(dev_in, in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(dev_out, 0, bytes);

    int blocks = 256;
    int grids = (n + blocks - 1) / blocks;
    float *d_exp, *d_partial;
    cudaMalloc(&d_exp, bytes);
    cudaMalloc(&d_partial, grids * 2 * sizeof(float));
    // 第一级：block→partial + block→exp

    softmax_block<<<grids, blocks>>>(dev_in, d_exp, d_partial, n);

    // 第二级：partial→global_max/global_sum + exp÷global_sum
    int blocks_2 = grids > 1024 ? 512 : grids;
    int smem = 2 * blocks_2 * sizeof(float); // 第二级只用 2*blocks 字节
    softmax_final<<<1, blocks_2, smem>>>(d_exp, d_partial, dev_out, blocks_2);
    cudaMemcpy(out, dev_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_exp);
    cudaFree(d_partial);
    cudaFree(dev_in);
    cudaFree(dev_out);
}

int main()
{
    int n = 1 << 20;
    auto input = random_vector(n, -100.0, 100.0);
    // auto start = std::chrono::system_clock::now();
    // auto it = std::max_element(input.begin(), input.end());
    // float *gold = new float[n];
    // softmax_cpu(input.data(), n, *it, gold);
    // auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
    //     std::chrono::system_clock::now() - start);
    // std::cout << "softmax op cost by CPU:" << cpu_duration.count() * 1e-3 << "ms"
    //           << std::endl;
    float *gpu_res = new float[n];
    softmax_func(input.data(), gpu_res, n);

    // if (check_vec_res(gpu_res, gold, n))
    // {
    //     std::cout << "Test Pass" << std::endl;
    // }
    // else
    // {
    //     std::cout << "Test Failed" << std::endl;
    // }

    delete[] gpu_res;
    // delete[] gold;
    return 0;
}