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

bool check_res(float a, float b)
{

    if (std::abs(a - b) > 1e-6)
    {
        return false;
    }

    return true;
}

bool check_vec_res(float *a, float *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (std::abs(a[i] - b[i]) > 0.0000001)
        {
            std::cerr << "i:" << i << " a[i]:" << a[i] << ",b[i]:" << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

__device__ float device_max(float a, float b) { return (a > b) ? a : b; }

__device__ float max_warp_shuffle(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = device_max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__global__ void softmax_step1_1(const float *in, float *out, int n)
{
    int tid = threadIdx.x;
    int id = tid + blockIdx.x * blockDim.x;
    float val = id < n ? in[id] : -device_infinity;
    val = max_warp_shuffle(val);
    extern __shared__ float shm[];
    if (tid % 32 == 0)
    {
        shm[tid / 32] = val;
    }
    __syncthreads();
    int warp_size = blockDim.x / 32;
    if (tid < warp_size)
    {
        val = shm[tid];
        val = max_warp_shuffle(val);
    }
    if (tid == 0)
    {
        out[blockIdx.x] = val;
    }
}

__global__ void softmax_step1_2(const float *in, float *out, int blocks)
{
    int tid = threadIdx.x;
    float res = -device_infinity;
    for (int i = tid; i < blocks; i += blockDim.x)
    {
        res = device_max(res, in[i]);
    }
    res = max_warp_shuffle(res);
    extern __shared__ float shm[];
    if (tid % 32 == 0)
    {
        shm[tid / 32] = res;
    }
    __syncthreads();
    int warp_size = blockDim.x / 32;
    if (tid < warp_size)
    {
        res = shm[tid];
        res = max_warp_shuffle(res);
    }
    if (tid == 0)
    {
        *out = res;
    }
}

inline __device__ float warpReduceSum(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__global__ void softmax_step2_0(const float *in, float *out, int n,
                                float *max_val)
{
    int tid = threadIdx.x;
    int id = tid + blockDim.x * blockIdx.x;
    float val = id < n ? expf(in[id] - *max_val) : 0.0f;
    if (id < n)
    {
        out[id] = val;
    }
}

__global__ void softmax_step2_1(const float *d_in, float *d_out, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    float val = (idx < n) ? d_in[idx] : 0;
    val = warpReduceSum(val); // 求一个warp内的和
    // 每个warp的首线程（lane 0）将结果存入共享内存，再做block内归约
    extern __shared__ float sdata[];
    if (tid % 32 == 0)
    {
        sdata[tid / 32] = val;
    }
    __syncthreads();
    // 块内归约
    int warp_size = blockDim.x / 32;
    if (tid < warp_size)
    {
        val = sdata[tid];
        val = warpReduceSum(val);
    }
    if (tid == 0)
    {
        d_out[blockIdx.x] = val;
    }
}

__global__ void softmax_step2_2(const float *in, float *out, int blocks)
{
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x)
    {
        local += in[i];
    }
    local = warpReduceSum(local);
    extern __shared__ float shm[];
    if (tid % 32 == 0)
    {
        shm[tid / 32] = local;
    }
    __syncthreads();
    int warp_size = blockDim.x / 32;
    if (tid < warp_size)
    {
        local = shm[tid];
        local = warpReduceSum(local);
    }
    if (tid == 0)
    {
        *out = local;
    }
}

__global__ void softmax_step3(const float *in, float *out, float *sum_value,
                              int n)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n)
    {
        out[id] = in[id] / (*sum_value);
    }
}

void softmax_func(const float *in, float *out, int n)
{
    cudaFree(0);
    float *dev_in, *dev_out;
    size_t bytes = n * sizeof(float);
    cudaMalloc(&dev_in, bytes);
    cudaMalloc(&dev_out, bytes);
    cudaMemcpy(dev_in, in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(dev_out, 0, bytes);
    float *dev_max, *dev_max_tmp;
    cudaMalloc(&dev_max, sizeof(float));
    cudaMemset(dev_max, 0, sizeof(float));
    int blocks = 256;
    int grids = (n + blocks - 1) / blocks;
    cudaMalloc(&dev_max_tmp, grids * sizeof(float));
    cudaMemset(dev_max_tmp, 0, grids * sizeof(float));

    // variable for step 2
    float *dev_sum, *dev_sum_tmp, *ele_tmp;
    cudaMalloc(&dev_sum, sizeof(float));
    cudaMemset(dev_sum, 0, sizeof(float));
    cudaMalloc(&dev_sum_tmp, grids * sizeof(float));
    cudaMemset(dev_sum_tmp, 0, grids * sizeof(float));
    cudaMalloc(&ele_tmp, n * sizeof(float));
    cudaMemset(ele_tmp, 0, n * sizeof(float));

    // step1: get the max value from the input
    auto start_time_1 = std::chrono::system_clock::now();
    softmax_step1_1<<<grids, blocks, blocks / 32 * sizeof(float)>>>(
        dev_in, dev_max_tmp, n);
    int blocks_step_2 = grids > 1024 ? 512 : grids;
    softmax_step1_2<<<1, blocks_step_2, blocks_step_2 / 32 * sizeof(float)>>>(
        dev_max_tmp, dev_max, grids);

    // if need check the max value
    // float *max_value = new float[1];
    // cudaDeviceSynchronize();
    // cudaMemcpy(max_value, dev_max, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "max value:" << *max_value << std::endl;
    // delete[] max_value;
    // end

    // step2 get each exp(item - max_value) & sum of all
    softmax_step2_0<<<grids, blocks>>>(dev_in, ele_tmp, n, dev_max);
    softmax_step2_1<<<grids, blocks, blocks / 32 * sizeof(float)>>>(
        ele_tmp, dev_sum_tmp, n);
    softmax_step2_2<<<1, blocks_step_2, blocks_step_2 / 32 * sizeof(float)>>>(
        dev_sum_tmp, dev_sum, grids);
    // if need check the sum value
    // auto end_time_2 = std::chrono::system_clock::now();
    // float *sum_value = new float[1];
    // cudaMemcpy(sum_value, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "sum value:" << *sum_value << std::endl;
    // delete[] sum_value;
    // end

    // step3: get the softmax result
    softmax_step3<<<grids, blocks>>>(ele_tmp, dev_out, dev_sum, n);
    auto end_time_3 = std::chrono::system_clock::now();
    auto gpu_sf_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time_3 - start_time_1);
    std::cout << "Your CUDA softmax op took: " << gpu_sf_duration.count() * 1e-3
              << "ms." << std::endl;
    cudaMemcpy(out, dev_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_max_tmp);
    cudaFree(dev_max);
    cudaFree(dev_sum_tmp);
    cudaFree(dev_sum);
    cudaFree(ele_tmp);
    cudaFree(dev_in);
    cudaFree(dev_out);
}

void softmax_cpu(const std::vector<float> &input, int n, float *res)
{
    float sum = 0.0f;
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