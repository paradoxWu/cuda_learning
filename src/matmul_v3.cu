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

__global__ void warm_up()
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel1st(float *dA, float *dB, float *dC, int M, int K,
                                int N)
{
    __shared__ float SA[BM][BK];
    __shared__ float SB[BK][BN];
    // 每一个thread 处理行 TM 和 列 TN 个数据
    // 因此计算A和B矩阵的全局坐标需要用global thread 坐标乘上TM /TN
    // 事实上这里的blockDim.x*TM就是BM, blockDim.y*TN就是BN
    // int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    // int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    // 等价写法,更好理解
    int block_m_start = blockIdx.x * BM;
    int block_n_start = blockIdx.y * BN;
    int thread_m_start = block_m_start + threadIdx.x * TM;
    int thread_n_start = block_n_start + threadIdx.y * TN;
    // 对A矩阵从列上进行分块，对B矩阵在行上进行分块, 都是K
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    // 拷贝到shared memory
    for (int ph = 0; ph < width; ph++)
    {
        // 当前K维度分块的全局起始偏移
        int k_block_start = ph * BK;
        for (int m = 0; m < TM; m++)
        {
            int gm = thread_m_start + m;
            for (int k = 0; k < BK; k++)
            {
                // 计算全局K维度索引
                int gk = k_block_start + k;
                if (gm < M && gk < K)
                {
                    SA[m + threadIdx.x * TM][k] = dA[gm * K + gk];
                }
                else
                {
                    SA[m + threadIdx.x * TM][k] = 0;
                }
            }
        }

        for (int n = 0; n < TN; n++)
        {
            int gn = thread_n_start + n;
            for (int k = 0; k < BK; k++)
            {
                // 计算全局K维度索引
                int gk = k_block_start + k;
                if (gn < N && gk < K)
                {
                    SB[k][threadIdx.y * TN + n] = dB[gk * N + gn];
                }
                else
                {
                    SB[k][threadIdx.y * TN + n] = 0.0f;
                }
            }
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k)
        {
            for (int m = 0; m < TM; ++m)
            {
                for (int n = 0; n < TN; ++n)
                {
                    int tmp_idx = m * TN + n;
                    tmp[tmp_idx] +=
                        SA[threadIdx.x * TM + m][k] * SB[k][threadIdx.y * TN + n];
                }
            }
        }
        __syncthreads();
    }

    // 寄存器结果写入全局内存dC（终极边界校验）=====================
    for (int m = 0; m < TM; ++m)
    {
        for (int n = 0; n < TN; ++n)
        {
            // 最终要写入的C矩阵全局坐标
            int gm = thread_m_start + m;
            int gn = thread_n_start + n;
            // 寄存器tmp一维索引
            int tmp_idx = m * TN + n;
            // 最终兜底校验：仅全局坐标有效时才写入，彻底杜绝非法内存访问
            if (gm < M && gn < N)
            {
                dC[gm * N + gn] = tmp[tmp_idx];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel2nd(float *dA, float *dB, float *dC, int M, int K,
                                int N)
{
    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;
    float tmp[TM * TN] = {0.0f};
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid % 128;
    int smem_a_k = tid / 128;
    int smem_b_k = tid % 8;
    int smem_b_n = tid / 8;
    for (int ph = 0; ph < width; ph++)
    {

        if (indA + smem_a_m < M && smem_a_k + ph * BK < K)
        {
            SA[smem_a_m * BK + smem_a_k] =
                dA[(indA + smem_a_m) * K + smem_a_k + ph * BK];
        }
        else
        {
            SA[smem_a_m * BK + smem_a_k] = 0.0f;
        }
        if (indB + smem_b_n < N && smem_b_k + ph * BK < K)
        {

            SB[smem_b_k * BN + smem_b_n] =
                dB[(smem_b_k + ph * BK) * N + indB + smem_b_n];
        }
        else
        {
            SB[smem_b_k * BN + smem_b_n] = 0.0f;
        }

        __syncthreads();
        for (int index_q = 0; index_q < TM; index_q++)
        {
            for (int index_v = 0; index_v < TN; index_v++)
            {
                int reg_c_m = threadIdx.x * TM + index_q;
                int reg_c_n = threadIdx.y * TN + index_v;
                for (int index_k = 0; index_k < BK; index_k++)
                {
                    tmp[index_q * TN + index_v] +=
                        SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n];
                }
            }
        }
        __syncthreads();
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v];
            }
        }
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
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int BLOCK_DIM_x = 16;
    constexpr int BLOCK_DIM_y = 16;
    constexpr int BM = TM * BLOCK_DIM_x;
    constexpr int BN = TN * BLOCK_DIM_y;
    constexpr int BK = 8;
    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid(num_blocks_x, num_blocks_y, 1);

    matrixKernel1st<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    // matrixKernel2nd<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, K,
    // N);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now() - start_time);
    std::cout << "cost by GPU v:" << duration.count() << "ms" << std::endl;
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
    warm_up<<< 1, 5 >>>();
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