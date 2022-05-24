#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n){
        C[i] = A[i]+B[i];
    }
}

void vecAddGpu(float* A, float* B, float* C, int n)
{
    //space to allocate on the device
    int size = n*sizeof(float);
    //pointer to pass in the allocate function
    float* A_d = NULL;
    float* B_d = NULL;
    float* C_d = NULL;  
    
    // Allocate device memory for A, B, and C
    // copy A and B to device memory
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Kernel launch code –to have the device
    // to perform the actual vector addition
    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;

    std::chrono::time_point<std::chrono::system_clock> start_time(std::chrono::system_clock::now());
    vecAddKernel<<<blockPerGrid,threadPerBlock>>>(A_d,B_d,C_d,n);
    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_time);
    std::cout<<"cost by GPU:"<<duration.count()<<"ms"<<std::endl;

    // copy C from the device memory
    // Free device vectors  
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    // Free device memory for A, B, C
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}

void vecAddCpu(float* A, float* B, float* C, int n)
{

    std::chrono::time_point<std::chrono::system_clock> start_time(std::chrono::system_clock::now());
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_time);
    std::cout<<"cost by CPU:"<<duration.count()<<"ms"<<std::endl;

}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    size_t size = n * sizeof(float);
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }
    vecAddCpu(a,b,c,n);
    vecAddGpu(a,b,c,n);

    free(a);
    free(b);
    free(c);
    return 0;
}