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

__global__ void reduce0(const float *in, float *out, int n)
{
    const int thread_size = blockDim.x;
    extern __shared__ float shm[]; // 动态共享内存
    int tid = threadIdx.x;
    int i = threadIdx.x;
    float local = 0.0f;
    // grid-stride loop 让每个线程处理多个元素
    while (i < n)
    {
        local += in[i];
        i += blockDim.x;
    }
    shm[tid] = local;
    __syncthreads();

    for (int step = 1; step < thread_size; step *= 2)
    {
        if (threadIdx.x % (2 * step) == 0)
        {
            shm[threadIdx.x] += shm[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (tid == 0)
        *out = shm[0];
}

// 交叉规约
__global__ void reduce_neighbor_1block(const float *g_in, float *g_out, int n)
{
    extern __shared__ float shm[]; // 动态共享内存
    int tid = threadIdx.x;

    /* 1. 初加载：grid-stride 累加，保证所有线程吃饱 */
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        local += g_in[i];
    shm[tid] = local;
    __syncthreads();

    /* 2. 相邻减半（neighbor / sequential） */
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();               // 先同步，再读
        if (tid < blockDim.x - stride) // 工作线程：0 .. (stride-1)
            shm[tid] += shm[tid + stride];
    }

    if (tid == 0)
        *g_out = shm[0];
}

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
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    shm[tid] = (i < blocks) ? g_in[i] : 0.0f;
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

    int *d_active_warp_count;    // 每个SM的活跃Warp数
    int *d_total_active_threads; // 总活跃线程数
    cudaMalloc(&d_active_warp_count, gpu_info.num_sms * sizeof(int));
    cudaMalloc(&d_total_active_threads, sizeof(int));
    cudaMemset(d_active_warp_count, 0, gpu_info.num_sms * sizeof(int));
    cudaMemset(d_total_active_threads, 0, sizeof(int));

    const int block_size = 256;
    int grid_size = 1;

    // warm-up
    reduce0<<<grid_size, block_size, block_size * sizeof(float)>>>(d_in, d_out,
                                                                   n);
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
            reduce0<<<grid_size, block_size, block_size * sizeof(float)>>>(d_in,
                                                                           d_out, n);
            break;
        case (1):
            reduce_neighbor_1block<<<grid_size, block_size,
                                     block_size * sizeof(float)>>>(d_in, d_out, n);
        case (2):
            grid_size = n / block_size;

            cudaMalloc(&d_tmp, sizeof(float) * grid_size);
            cudaMemset(d_tmp, 0, sizeof(float) * grid_size);

            reduce_interleaved<<<grid_size, block_size, block_size * sizeof(float)>>>(
                d_in, d_tmp, n);
            reduce_interleaved_final<<<1, grid_size, grid_size * sizeof(float)>>>(
                d_tmp, d_out, n);
            cudaFree(d_tmp);
            break;
        case (3):
            grid_size = n / block_size;
            cudaMalloc(&d_tmp, sizeof(float) * grid_size);
            cudaMemset(d_tmp, 0, sizeof(float) * grid_size);

            reduce_neighbor<<<grid_size, block_size, block_size * sizeof(float)>>>(
                d_in, d_tmp, n);
            reduce_neighbor_final<<<1, grid_size, grid_size * sizeof(float)>>>(
                d_tmp, d_out, n);
            cudaFree(d_tmp);
            break;
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

    // TODO 2：用 __activemask() 和 __popc() 统计实际活跃 warp 数，计算 occupancy
    std::vector<int> h_active_warp_count(gpu_info.num_sms);
    int h_total_active_threads = 0;
    cudaMemcpy(h_active_warp_count.data(), d_active_warp_count,
               gpu_info.num_sms * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_active_threads, d_total_active_threads, sizeof(int),
               cudaMemcpyDeviceToHost);
    // 计算实际Occupancy
    int total_active_warps = 0;
    for (int sm = 0; sm < gpu_info.num_sms; sm++)
    {
        total_active_warps += h_active_warp_count[sm];
    }
    // 平均每个SM的活跃Warp数
    float avg_active_warps_per_sm = (float)total_active_warps / gpu_info.num_sms;
    // 实际Occupancy（%）
    float actual_occupancy =
        (avg_active_warps_per_sm / gpu_info.max_warps_per_sm) * 100.0f;
    // 活跃线程占比
    float active_thread_ratio =
        (float)h_total_active_threads / (grid_size * block_size) * 100.0f;
    // 输出结果
    // std::cout
    //     << "\n===================== Occupancy 统计结果 ====================="
    //     << std::endl;
    // std::cout << "核函数配置: 网格大小=" << grid_size
    //           << ", 线程块大小=" << block_size << std::endl;
    // std::cout << "总活跃Warp数: " << total_active_warps << std::endl;
    // std::cout << "平均每个SM活跃Warp数: " << avg_active_warps_per_sm << std::endl;
    // std::cout << "实际Occupancy: " << actual_occupancy << "%" << std::endl;
    // std::cout << "总活跃线程数: " << h_total_active_threads << " / "
    //           << grid_size * block_size << std::endl;
    // std::cout << "活跃线程占比: " << active_thread_ratio << "%" << std::endl;
    // // 释放资源
    cudaFree(d_active_warp_count);
    cudaFree(d_total_active_threads);
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
    // TODO 3：把 reduce1 改成 "完全 warp shuffle 无共享内存" 版本，再测一次带宽

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
