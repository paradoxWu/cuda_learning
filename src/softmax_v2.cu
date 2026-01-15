#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <cmath>
#include <cub/cub.cuh>
const float device_infinity = HUGE_VALF;
#define BLOCK_SIZE 256
#define WARP_SIZE 32

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

struct __align__(8) MD_F
{
    float m; // 最大值
    float d; // 指数和
};

// “合并”两个 MD_F 结构，得到同时代表这两部分数据的最大值和归一化因子。其核心思路
__device__ __forceinline__ MD_F reduce_md_op(MD_F a, MD_F b)
{
    bool a_bigger = (a.m > b.m);
    MD_F bigger_m = a_bigger ? a : b;
    MD_F smaller_m = a_bigger ? b : a;
    MD_F res;
    // dj = exp^(xj-mj)+dj-1 * exp^(mj-1 - mj)
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

__global__ void onlineSoftmaxKernel(const float *__restrict__ mat,
                                    float *__restrict__ output,
                                    int ncol)
{
    MD_F mdf_val = {-1e20f, 0.0f}, mdf_tmp;
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        mdf_tmp.m = mat[blockIdx.x * ncol + i];
        mdf_tmp.d = 1.0f;
        mdf_val = reduce_md_op(mdf_tmp, mdf_val);
    }
    typedef cub::BlockReduce<MD_F, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ MD_F mdf_total;
    mdf_val = BlockReduce(tempStorage).Reduce(mdf_val, reduce_md_op);
    if (threadIdx.x == 0)
        mdf_total = mdf_val;
    __syncthreads();
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
    {
        output[blockIdx.x * ncol + i] =
            __expf(mat[blockIdx.x * ncol + i] - mdf_total.m) / mdf_total.d;
    }
}

template <int THREADBLOCK_SIZE>
__global__ void online_softmax(
    const float *__restrict x,
    float *__restrict y,
    int V)
{
    int tid = threadIdx.x;
    int vector_id = blockIdx.x;

    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD_F, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD_F md_total;

    // 每个线程计算局部MD值
    MD_F md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0f;
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        MD_F new_elem;
        new_elem.m = x[elem_id]; // 一次遍历数据即可完成d,m的动态更新 相比原始softmax 有效降低访存压力 潜在增加了缓存的利用率
        new_elem.d = 1.0f;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    MD_F md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (tid == 0)
    {
        md_total = md;
    }
    __syncthreads();

    float d_total_inverse = __fdividef(1.0f, md_total.d);
    for (int elem_id = tid; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
    }
}

__global__ void optimized_softmax(float *input, float *output, int n, float *global_sum)
{
    __shared__ float shared_data[BLOCK_SIZE];
    __shared__ float shared_exp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    // 加载数据
    float val = (idx < n) ? input[idx] : -device_infinity;
    shared_data[tid] = val;
    __syncthreads();
    // 找块内最大值
    float max_val = shared_data[0];
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            max_val = fmaxf(max_val, shared_data[tid + s]);
        }
        __syncthreads();
        if (s == 1)
            shared_data[tid] = max_val;
        __syncthreads();
    }
    max_val = shared_data[0];
    // 计算指数（避免银行冲突）
    int offset = tid % WARP_SIZE;
    for (int i = offset; i < BLOCK_SIZE; i += WARP_SIZE)
    {
        shared_exp[i] = expf(shared_data[i] - max_val);
    }
    __syncthreads();
    // 块内归约
    float sum = 0.0f;
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_exp[tid] += shared_exp[tid + s];
        }
        __syncthreads();
        if (s == 1)
            sum = shared_exp[tid];
    }
    // 原子累加全局和
    if (tid == 0)
    {
        atomicAdd(global_sum, sum);
    }
    __syncthreads();
    // 归一化（需等待全局和就绪）
    float total_sum = *global_sum;
    if (idx < n)
    {
        output[idx] = shared_exp[tid] / total_sum;
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

    int blocks = BLOCK_SIZE;
    int grids = (n + blocks - 1) / blocks;

    auto start_time_1 = std::chrono::system_clock::now();
    onlineSoftmaxKernel<<<grids, blocks>>>(dev_in, dev_out, n);
    // optimized_softmax<<<grids, blocks>>>(dev_in, dev_out, n, d_sum);
    // online_softmax<256><<<grids, blocks>>>(dev_in, dev_out, n);
    auto end_time_3 = std::chrono::system_clock::now();
    auto gpu_sf_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_3 - start_time_1);
    std::cout << "Your CUDA softmax op took: " << gpu_sf_duration.count() * 1e-3
              << "ms." << std::endl;
    cudaMemcpy(out, dev_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dev_in);
    cudaFree(dev_out);
}

void softmax_cpu(const std::vector<float> &input, int n, float *res)
{
    float maxVal = *std::max_element(input.begin(), input.end()); // 最大值平移防溢出
    std::cout << "total max value by cpu:" << maxVal << std::endl;
    std::vector<float> expValues(input.size());
    for (size_t i = 0; i < input.size(); ++i)
    {
        expValues[i] = std::exp(input[i] - maxVal);
    }

    float sumExp = std::accumulate(expValues.begin(), expValues.end(), 0.0);
    std::cout << "total sum by cpu:" << sumExp << std::endl;
    for (size_t i = 0; i < input.size(); ++i)
    {
        res[i] = expValues[i] / sumExp;
    }
}

bool check_vec_res(float *a, float *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (std::abs(a[i] - b[i]) > 1e-6)
        {
            std::cerr << "i:" << i << " a[i]:" << a[i] << ",b[i]:" << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    int n = 1 << 20;
    auto input = random_vector(n, -100.0, 100.0);
    auto start = std::chrono::system_clock::now();
    float *gold = new float[n];
    softmax_cpu(input, n, gold);
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start);
    std::cout << "softmax op cost by CPU:" << cpu_duration.count() * 1e-3 << "ms"
              << std::endl;
    float *gpu_res = new float[n];
    softmax_func(input.data(), gpu_res, n);

    if (check_vec_res(gpu_res, gold, n))
    {
        std::cout << "Test Pass" << std::endl;
    }
    else
    {
        std::cout << "Test Failed" << std::endl;
    }

    delete[] gpu_res;
    delete[] gold;
    return 0;
}