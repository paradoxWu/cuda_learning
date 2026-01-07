// gauss_cpu.cpp
//g++ -O3 -fopenmp gaussian_blur.cpp -o omp && ./omp
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>


void omp() {
  const int H = 1080, W = 1920;
  const float sigma = 3.0f;
  int radius = static_cast<int>(std::ceil(3.0f * sigma));
  int k_size = 2 * radius + 1;
  std::vector<float> kernel(k_size);
  float sum = 0.0f;
  for (int i = 0; i < k_size; ++i) {
    float x = i - radius;
    kernel[i] = std::exp(-0.5f * x * x / (sigma * sigma));
    sum += kernel[i];
  }
  for (auto &v : kernel)
    v /= sum;

  std::vector<float> img(H * W, 1.0f), tmp(H * W), dst(H * W);

  auto t0 = std::chrono::high_resolution_clock::now();

// 横向
#pragma omp parallel for
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float sum = 0.0f;
      for (int k = 0; k < k_size; ++k) {
        int sx = x - radius + k;
        sx = std::max(0, std::min(sx, W - 1));
        sum += img[y * W + sx] * kernel[k];
      }
      tmp[y * W + x] = sum;
    }
  }
// 纵向
#pragma omp parallel for
  for (int x = 0; x < W; ++x) {
    for (int y = 0; y < H; ++y) {
      float sum = 0.0f;
      for (int k = 0; k < k_size; ++k) {
        int sy = y - radius + k;
        sy = std::max(0, std::min(sy, H - 1));
        sum += tmp[sy * W + x] * kernel[k];
      }
      dst[y * W + x] = sum;
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "CPU OpenMP: " << ms << " ms\n";
}

void singthread() {
  const int H = 1080, W = 1920;
  const float sigma = 3.0f;
  int radius = static_cast<int>(std::ceil(3.0f * sigma));
  int k_size = 2 * radius + 1;

  // 1. 生成 1D 高斯核
  std::vector<float> kernel(k_size);
  float sum = 0.0f;
  for (int i = 0; i < k_size; ++i) {
    float x = i - radius;
    kernel[i] = std::exp(-0.5f * x * x / (sigma * sigma));
    sum += kernel[i];
  }
  for (auto &v : kernel)
    v /= sum;

  // 2. 造图
  std::vector<float> img(H * W, 1.0f), tmp(H * W), dst(H * W);

  // 3. 横向卷积
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float sum = 0.0f;
      for (int k = 0; k < k_size; ++k) {
        int sx = x - radius + k;
        sx = std::max(0, std::min(sx, W - 1));
        sum += img[y * W + sx] * kernel[k];
      }
      tmp[y * W + x] = sum;
    }
  }
  // 4. 纵向卷积
  for (int x = 0; x < W; ++x) {
    for (int y = 0; y < H; ++y) {
      float sum = 0.0f;
      for (int k = 0; k < k_size; ++k) {
        int sy = y - radius + k;
        sy = std::max(0, std::min(sy, H - 1));
        sum += tmp[sy * W + x] * kernel[k];
      }
      dst[y * W + x] = sum;
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "CPU single: " << ms << " ms\n";
}

void gaussian_thread() {
  const int H = 1080, W = 1920;
  const float sigma = 3.0f;
  int radius = static_cast<int>(std::ceil(3.0f * sigma));
  int k_size = 2 * radius + 1;
  auto size_2 = k_size >> 1;

  // 1. 生成 1D 高斯核
  std::vector<float> kernel(k_size);
  float sum = 0.0f;
  for (int i = 0; i < k_size; ++i) {
    float x = i - radius;
    kernel[i] = std::exp(-0.5f * x * x / (sigma * sigma));
    sum += kernel[i];
  }
  for (auto &v : kernel)
    v /= sum;

  // 2. 造图
  std::vector<float> img(H * W, 1.0f), tmp(H * W), dst(H * W);

  const auto max_thread = std::thread::hardware_concurrency();
  std::vector<std::thread> thread_bar;
  const auto t_rows = H / (max_thread);
  // initialize the gaussian window

  auto compGassion_thread = [&](const int thread_id) {
    for (auto i{t_rows * (thread_id - 1)}; i < t_rows * thread_id; ++i) {
      for (auto j{size_2}; j < W - size_2; ++j) {
        double sum = 0.0;
        for (size_t y = 0; y < k_size; ++y) {
          for (size_t x = 0; x < k_size; ++x) {
            sum += (img[(i + y) * W + j + x] * kernel[x * k_size + y]);
          }
        }
        dst[i * W + j] = static_cast<char>(sum);
      }
    }
  };

  std::chrono::time_point<std::chrono::system_clock> start_time(
      std::chrono::system_clock::now());
  for (int thread_id = 1; thread_id <= max_thread; ++thread_id) {
    thread_bar.emplace_back(compGassion_thread, thread_id);
  }
  for (auto &i : thread_bar) {
    i.join();
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now() - start_time);
  std::cout << "cost by CPU_Concurrency:" << duration.count() << "ms"
            << std::endl;
}

int main() {
  singthread();
  omp();
  gaussian_thread();
  return 0;
}