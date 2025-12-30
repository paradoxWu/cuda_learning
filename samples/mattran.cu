#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

bool GetDif(int M, int N, const float *mat1, const float *mat2)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (mat1[i * N + j] - mat2[i * N + j] >
                std::numeric_limits<float>::epsilon())
            {
                std::cout << "error happen:" << i << "," << j
                          << ": " << mat1[i * N + j] << "  " << mat2[i * N + j];
                return false;
            }
        }
    }
    return true;
}

void showMatrix(int M, int N, const float *mat)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            std::cout << mat[i * N + j] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void matrixTranCpu(const std::vector<float> &in, std::vector<float> &out, int M,
                   int N)
{
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out[j * M + i] = in[i * N + j];
        }
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by CPU:" << duration.count() << "ms" << std::endl;
}

__global__ void mattranGpu_v1(const float *in, float *out, int M, int N)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col < N && row < M)
    {
        out[col * M + row] = in[row * N + col];
    }
}

template <int BLOCK_DIM>
__global__ void mattranGpu_v2(const float *in, float *out, int M, int N)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    __shared__ float SA[BLOCK_DIM][BLOCK_DIM];
    // copy to shared memory
    bool valid = (col < N) && (row < M);
    if (valid)
    {
        SA[threadIdx.y][threadIdx.x] = in[row * N + col];
    }
    __syncthreads();
    // copy to out matrix
    int out_x = threadIdx.x + blockIdx.y * BLOCK_DIM;
    int out_y = threadIdx.y + blockIdx.x * BLOCK_DIM;
    if (valid)
        out[out_y * M + out_x] = SA[threadIdx.x][threadIdx.y];
}

void mattranGpu(const float *in, float *out, int M, int N)
{
    // malloc device memory
    float *dev_in;
    float *dev_out;
    size_t space_size = M * N * sizeof(float);
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    CUDA_CHECK(cudaMalloc(&dev_in, space_size));
    CUDA_CHECK(cudaMalloc(&dev_out, space_size));
    // copy to device
    CUDA_CHECK(cudaMemcpy(dev_in, in, space_size, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    // call kernel function
    // mattranGpu_v1<<<grid, block>>>(dev_in, dev_out, M, N);
    mattranGpu_v2<16><<<grid, block>>>(dev_in, dev_out, M, N);
    // copy result to host
    CUDA_CHECK(cudaMemcpy(out, dev_out, space_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_in));
    CUDA_CHECK(cudaFree(dev_out));
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by GPU:" << duration.count() << "ms" << std::endl;
}

int main()
{
    const int M = 1024, N = 512;

    // Host memory
    std::vector<float> h_A(M * N, 0.0f);
    std::vector<float> h_B(M * N, 0.0f);
    std::vector<float> h_C(M * N, 0.0f);
    for (int i = 0; i < M * N; i++)
    {
        h_A[i] = i + 1;
    }
    // cpu calculate
    matrixTranCpu(h_A, h_B, M, N);
    // showMatrix(M, N, h_B.data());
    mattranGpu(h_A.data(), h_C.data(), M, N);
    if (GetDif(M, N, h_B.data(), h_C.data()))
    {
        std::cout << "Test Pass" << std::endl;
    }
    else
    {
        std::cout << "Test Failed" << std::endl;
    }

    return 0;
}