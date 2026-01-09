#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <torch/script.h>

// 假设你已经定义了cudaReduceSum函数
float cudaReduceSum(float *data, int size)
{
}

int main()
{
    // 创建一个随机张量
    int n = 1 << 20;
    torch::Tensor tensor =
        torch::rand({n}, torch::kFloat32).to(torch::kCUDA);
    float *data = tensor.data_ptr<float>();
    int size = tensor.numel();

    // 调用自己的CUDA归约算子
    // auto start = std::chrono::high_resolution_clock::now();
    // float yourResult = cudaReduceSum(data, size);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> yourDuration = end - start;
    // std::cout << "Your CUDA reduce took: " << yourDuration.count() << "
    // seconds."
    // << std::endl;
    
    // 调用LibTorch的sum算子
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor libtorchResult = torch::sum(tensor);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> libtorchDuration = end - start;

    std::cout << "LibTorch sum took: " << libtorchDuration.count() << " seconds."
              << std::endl;
    // 验证结果
    // if (std::abs(yourResult - libtorchResult.item<float>()) > 1e-6) {
    // std::cerr << "Results do not match!" << std::endl;
    // } else {
    // std::cout << "Results match." << std::endl;
    // }

    // 输出耗时

    return 0;
}