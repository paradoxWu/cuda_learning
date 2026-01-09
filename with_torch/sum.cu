#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <torch/script.h>

bool check_res(float a, float b)
{

    if (std::abs(a - b) > 1e-6)
    {

        printf("error happen: a: %.3f, b: %3f\n", a, b);
        return false;
    }

    return true;
}

__device__ float warp_shuffle(float val)
{
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__global__ void reduce_final(const float *in, float *out, int blocks)
{
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < blocks; i += blockDim.x)
    {
        local += in[i];
    }
    local = warp_shuffle(local);
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
        local = warp_shuffle(local);
    }
    if (tid == 0)
    {
        *out = local;
    }
}
__global__ void reduce_sum(const float *in, float *out, int n)
{
    int tid = threadIdx.x;
    int id = tid + blockIdx.x * blockDim.x;
    float val = id < n ? in[id] : 0.0f;
    val = warp_shuffle(val);
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
        val = warp_shuffle(val);
    }
    if (tid == 0)
    {
        out[blockIdx.x] = val;
    }
}

float cudaReduceSum(float *data, int size)
{
    float *res = new float[1];
    float *d_in, *d_out;
    size_t bytes = size * sizeof(float);
    auto start_time_0 = std::chrono::system_clock::now();
    cudaEvent_t start, stop, ker_start, ker_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&ker_start);
    cudaEventCreate(&ker_stop);
    cudaEventRecord(start, 0);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, data, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));
    // warm up
    reduce_final<<<1, 256, 256 * sizeof(float)>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
    auto start_time_1 = std::chrono::system_clock::now();
    int thread_num = 256;
    int block_num = (size + thread_num - 1) / thread_num;
    float *d_tmp;
    cudaMalloc(&d_tmp, block_num * sizeof(float));
    cudaMemset(d_tmp, 0, sizeof(float) * block_num);
    // 只测 kernel
    cudaEventRecord(ker_start, 0);
    reduce_sum<<<block_num, thread_num, thread_num / 32 * sizeof(float)>>>(d_in, d_tmp, size);
    reduce_final<<<1, 512, 512 / 32 * sizeof(float)>>>(d_tmp, d_out, block_num);
    cudaEventRecord(ker_stop, 0);
    cudaEventSynchronize(ker_stop);
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, ker_start, ker_stop);
    printf("kernel only = %.3f ms\n", kernel_ms);
    cudaMemcpy(res, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    printf("kennel + memcpy  = %.3f ms\n", total_ms);
    auto end_time_1 = std::chrono::system_clock::now();
    cudaFree(d_tmp);
    cudaFree(d_in);
    cudaFree(d_out);
    float value = *res;
    delete[] res;
    auto end_time_0 = std::chrono::system_clock::now();
    auto yourDuration_0 = std::chrono::duration_cast<std::chrono::microseconds>(end_time_0 - start_time_0);
    auto yourDuration_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_time_1 - start_time_1);
    std::cout << "Your CUDA total took: " << yourDuration_0.count() * 1e-3 << "ms." << std::endl;
    std::cout << "Your CUDA reduce took: " << yourDuration_1.count() * 1e-3 << "ms." << std::endl;
    return value;
}

void reduce_cpu(const float *x, int n, float &res)
{
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
        sum += x[i];
    res = sum;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by CPU:" << duration.count() * 1e-3 << "ms" << std::endl;
}

int main()
{
    // 创建一个随机张量
    int n = 1 << 20;
    static bool warmed = false;
    if (!warmed)
    {
        cudaFree(0);                                              // CUDA context
        torch::sum(torch::rand({1}, torch::kCUDA)).item<float>(); // libtorch JIT
        warmed = true;
    }
    torch::Tensor tensor = torch::rand({n}, torch::kFloat32);

    float *data = tensor.data_ptr<float>();
    // float gold = 0.0f;
    // reduce_cpu(data, n, gold);
    // 调用自己的CUDA归约算子
    float yourResult = cudaReduceSum(data, n);

    // 调用LibTorch的sum算子
    tensor = tensor.to(torch::kCUDA);
    auto start = std::chrono::system_clock::now();
    torch::Tensor libtorchResult = torch::sum(tensor);
    float libtorch_sum = libtorchResult.item<float>();
    auto libtorchDuration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start);

    std::cout << "LibTorch sum took: " << libtorchDuration.count() * 1e-3 << " ms."
              << std::endl;
    // 验证结果
    // if (check_res(yourResult, gold))
    // {
    //     std::cout << "CPU Test Pass" << std::endl;
    // }
    // else
    // {
    //     std::cout << "CPU Test Failed" << std::endl;
    // }
    if (check_res(yourResult, libtorch_sum))
    {
        std::cout << "Torch Test Pass" << std::endl;
    }
    else
    {
        std::cout << " Torch Test Failed" << std::endl;
    }
    // if (check_res(gold, libtorch_sum))
    // {
    //     std::cout << "cpu torch is the same" << std::endl;
    // }
    // else
    // {
    //     std::cout << "cpu torch is dif" << std::endl;
    // }

    // 输出耗时

    return 0;
}