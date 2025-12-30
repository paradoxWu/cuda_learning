#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <vector>
/*
A: [M,k]
B: [K,N]
C: [M,N]
C = A*B
*/
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

bool GetDif(int M, int N, const float *mat1, const float *mat2)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (mat1[i * N + j] - mat2[i * N + j] >
                std::numeric_limits<float>::epsilon())
            {
                std::cout << "error happen:" << i << ":" << mat1[i * N + j] << "," << j
                          << ":" << mat2[i * N + j];
                return false;
            }
        }
    }
    return true;
}

void matrixCpu(const float *A, const float *B, float *C, int M, int N, int K)
{
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int k = 0; k < K; k++)
            {
                tmp += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = tmp;
        }
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by CPU:" << duration.count() << "ms" << std::endl;
    // showMatrix(M, N, C);
}

// 纯全局内存矩阵乘法（无共享内存）

__global__ void matrix_mul_naive(const float *A, const float *B, float *C,
                                 int M, int N, int K)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < M && col < N)
    {
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 共享内存加速
// warp读取shared memory减少数据重复读写
// 如果K太大，那么share memory空间不够，需要分段加载
template <int BLOCK_DIM>
__global__ void matrix_mul_v2(const float *A, const float *B, float *C, int M,
                              int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp = 0.0f;
    __shared__ float SA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float SB[BLOCK_DIM][BLOCK_DIM];
    int width = (K + BLOCK_DIM - 1) / BLOCK_DIM;
    for (int ph = 0; ph < width; ph++)
    {
        // memcpy to the share memory
        if (row < M && threadIdx.x + ph * BLOCK_DIM < K)
        {
            SA[threadIdx.y][threadIdx.x] = A[row * K + threadIdx.x + ph * BLOCK_DIM];
        }
        else
        {
            SA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (threadIdx.x + ph * BLOCK_DIM < K && col < N)
        {
            SB[threadIdx.y][threadIdx.x] =
                B[(threadIdx.y + ph * BLOCK_DIM) * N + col];
        }
        else
        {
            SB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int s = 0; s < BLOCK_DIM; s++)
        {
            tmp += SA[threadIdx.y][s] * SB[s][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = tmp;
    }
}

void matmulGpu(std::vector<float> h_A, std::vector<float> h_B,
               std::vector<float> &h_C, int M, int N, int K)
{
    // Device memory
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    std::chrono::time_point<std::chrono::system_clock> start_time(
        std::chrono::system_clock::now());
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    dim3 block(16, 16);
    dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);

    // matrix_mul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    matrix_mul_v2<16><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by GPU:" << duration.count() << "ms" << std::endl;
    // showMatrix(M, N, h_C.data());
}

int main()
{
    const int M = 512, N = 512, K = 512;

    // Host memory
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_D(M * N, 0.0f);
    // cpu calculate
    matrixCpu(h_A.data(), h_B.data(), h_C.data(), M, N, K);
    matmulGpu(h_A, h_B, h_D, M, N, K);
    if (GetDif(M, N, h_C.data(), h_D.data()))
    {
        std::cout << "Test Pass" << std::endl;
    }
    else
    {
        std::cout << "Test Failed" << std::endl;
    }

    return 0;
}