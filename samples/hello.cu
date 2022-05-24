#include <iostream>

__global__ void hello_world(){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[ %d ] hello world\n", idx);
}

int main() {
    hello_world<<< 1, 5 >>>();
    cudaDeviceSynchronize();
    return 0;
}