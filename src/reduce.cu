#include <assert.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <vector>

struct GPUInfo
{
    int max_warps_per_sm; // 每个SM支持的最大Warp数
    int threads_per_warp; // 每个Warp的线程数（固定32）
    int num_sms;          // GPU的SM总数
};

GPUInfo get_gpu_info(int device_id = 0)
{
    GPUInfo info;
    info.threads_per_warp = 32; // CUDA Warp固定32线程

    // 查询SM总数
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    info.num_sms = prop.multiProcessorCount;

    // 查询每个SM支持的最大Warp数（不同架构值不同）
    switch (prop.major * 10 + prop.minor)
    {
    case 80: // A100 (sm_80)
    case 86: // RTX 3090 (sm_86)
        info.max_warps_per_sm = 64;
        break;
    case 89: // RTX 4090 (sm_89)
        info.max_warps_per_sm = 64;
        break;
    case 75: // T4 (sm_75)
        info.max_warps_per_sm = 64;
        break;
    case 70: // V100 (sm_70)
        info.max_warps_per_sm = 64;
        break;
    case 61: // GTX 1080 (sm_61)
        info.max_warps_per_sm = 64;
        break;
    default: // 其他架构默认64
        info.max_warps_per_sm = 64;
        break;
    }
    return info;
}

//-------------------- 比较差异 --------------------
bool check_res(float a, float b)
{

    if (std::abs(a - b) > std::numeric_limits<float>::epsilon())
    {
        std::cout << "error happen:"
                  << "a:" << a << ", b:" << b;
        return false;
    }

    return true;
}

// -------------------- CPU 实现 --------------------
void reduce_cpu(const float *x, int n, float &res)
{
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    float sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += x[i];
    res = sum;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by CPU:" << duration.count() * 1e-3 << "ms" << std::endl;
}

// 交叉规约
__global__ void reduce_neighbor(const float *g_in, float *g_out, int n)
{
    extern __shared__ float shm[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    shm[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {                               // 相邻配对
        int idx = 2 * stride * tid; // 0,2,4,8...
        if (idx < blockDim.x)
        {
            shm[idx] += shm[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_out[blockIdx.x] = shm[0];
}

__global__ void reduce_neighbor_final(const float *g_in, float *g_out,
                                      int blocks)
{
    extern __shared__ float shm[];
    int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x)
    {
        val += g_in[i];
    }
    shm[tid] = val;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {                               // 相邻配对
        int idx = 2 * stride * tid; // 0,2,4,8...
        if (idx < blockDim.x)
        {
            shm[idx] += shm[idx + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        *g_out = shm[0];
}
// 交叉规约 end

// 交错规约
__global__ void reduce_interleaved(const float *g_in, float *g_out, int n)
{
    extern __shared__ float shm[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x; // 一人吃 2 元素
    float sum = 0.0f;
    if (i < n)
        sum += g_in[i];
    if (i + blockDim.x < n)
        sum += g_in[i + blockDim.x];
    shm[tid] = sum;
    __syncthreads();

    // 经典交错树
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            shm[tid] += shm[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        g_out[blockIdx.x] = shm[0];
}

__global__ void reduce_interleaved_final(const float *partial, float *out,
                                         int blocks)
{
    extern __shared__ float shm[];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x)
        local += partial[i];
    shm[tid] = local;
    __syncthreads();
    // 交错规约收尾
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            shm[tid] += shm[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        *out = shm[0];
}
// 交错规约 end

// shuffle warp
inline __device__ float warpReduceSum(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val; // 返回后 **所有线程** 都是 sum
}

__global__ void reduce_shuffle(const float *d_in, float *d_out, int n)
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

__global__ void reduce_shuffle_final(const float *d_in, float *d_out,
                                     int blocks)
{
    int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x)
    {
        val += d_in[i];
    }
    val = warpReduceSum(val);
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
        *d_out = val;
    }
}

// shuffle end
// cuda main func
void reducegpu(const float *h_in, float *h_out, int n, int method)
{
    // set cuda device info
    int device_id = 0;
    cudaSetDevice(device_id);
    GPUInfo gpu_info = get_gpu_info(device_id);

    // cuda malloc
    size_t bytes = n * sizeof(float);
    float *d_in, *d_out;
    float *d_tmp;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    const int block_size = 256;
    int grid_size = 1;

    // warm-up
    reduce_interleaved_final<<<grid_size, block_size,
                               block_size * sizeof(float)>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // 计时 10 次取平均
    int repeat = 10;
    for (int rep = 0; rep < repeat; ++rep)
    {
        switch (method)
        {
        case (0):
            reduce_interleaved_final<<<grid_size, block_size,
                                       block_size * sizeof(float)>>>(d_in, d_out, n);
            break;
        case (1):
            reduce_neighbor_final<<<grid_size, block_size,
                                    block_size * sizeof(float)>>>(d_in, d_out, n);
            break;
        case (2):
        {
            int blocks = n / block_size;
            cudaMalloc(&d_tmp, sizeof(float) * blocks);
            cudaMemset(d_tmp, 0, sizeof(float) * blocks);

            reduce_interleaved<<<blocks, block_size, block_size * sizeof(float)>>>(
                d_in, d_tmp, n);
            grid_size = blocks > 1024 ? 512 : blocks;
            reduce_interleaved_final<<<1, grid_size, grid_size * sizeof(float)>>>(
                d_tmp, d_out, blocks);
            cudaFree(d_tmp);
            break;
        }
        case (3):
        {
            int blocks = n / block_size;
            cudaMalloc(&d_tmp, sizeof(float) * blocks);
            cudaMemset(d_tmp, 0, sizeof(float) * blocks);

            reduce_neighbor<<<blocks, block_size, block_size * sizeof(float)>>>(
                d_in, d_tmp, n);
            grid_size = blocks > 1024 ? 512 : blocks;
            reduce_neighbor_final<<<1, grid_size, grid_size * sizeof(float)>>>(
                d_tmp, d_out, blocks);
            cudaFree(d_tmp);
            break;
        }
        case (4):
        {
            int blocks = n / block_size;
            cudaMalloc(&d_tmp, sizeof(float) * blocks);
            cudaMemset(d_tmp, 0, sizeof(float) * blocks);
            reduce_shuffle<<<blocks, block_size, block_size / 32 * sizeof(float)>>>(
                d_in, d_tmp, n);
            grid_size = blocks > 1024 ? 512 : blocks;
            reduce_shuffle_final<<<1, grid_size, grid_size / 32 * sizeof(float)>>>(
                d_tmp, d_out, blocks);
            cudaFree(d_tmp);
            break;
        }
        default:
            break;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    cudaDeviceSynchronize();
    double avg_elapsed_ms = ker_time / repeat;
    double ms = (duration.count() * 1e-3) / repeat;
    printf(" GPU use time:%.4f ms, kernel time:%.4f ms\n", ms, avg_elapsed_ms);

    // 计算L2 Read Throughput
    // TODO 1：打印 L2 throughput（用 cudaEventElapsedTime 估算即可）
    double l2_throughput_gbs = bytes / (ms * 1e-3) * 1e-9;
    printf("N=%d  time=%.3f ms  DRAM bandwidth=%.1f GB/s\n", n, ms,
           l2_throughput_gbs);

    cudaFree(d_in);
    cudaFree(d_out);
}

// -------------------- 主流程 --------------------
int main(int argc, char **argv)
{
    int n = 1 << 20;
    int method_id = (argc > 1) ? atoi(argv[1]) : 0;
    //   assert(n > 0 && (n & (n - 1)) == 0 && "length must be power-of-2");

    float *h_in, *h_out;

    h_in = new float[n];
    h_out = new float[1];
    for (int i = 0; i < n; ++i)
        h_in[i] = 1.0f;
    float gold = 0.0f;
    reduce_cpu(h_in, n, gold);
    reducegpu(h_in, h_out, n, method_id);

    // check the result
    if (check_res(h_out[0], gold))
    {
        std::cout << "Test Pass" << std::endl;
    }
    else
    {
        std::cout << "Test Failed" << std::endl;
    }

    delete[] h_in;
    delete[] h_out;
    return 0;
}