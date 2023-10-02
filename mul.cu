// #include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>
#include "mul.h"

// const int N = 1024; // Matrix size (N x N)

// used to define functions that are called from the CPU (host) but run on the GPU (device).
// also known as kernel functions
__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row*k + col] = sum;
    }
}

extern "C"
__global__ void Mul(float * A, float * B, float * C,
                   int A_shape_0,int A_shape_1,int B_shape_1) {
    float cValue = 0;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < A_shape_0) && (Col < B_shape_1)) {
        for (int k = 0; k < A_shape_1; k++) {
            cValue += A[Row*A_shape_1 + k] * B[k*B_shape_1 + Col];
        }
        C[Row*B_shape_1 + Col] = cValue;
    }
}

// Export C function to be called from Python
/*
warning: still don't work and I don't know why?
*/
// int main(){
extern "C" void Multrix(float* h_A, float* h_B, float* h_C, int m, int n, int k){
    // ... (rest of your CUDA function code)
    // int m = 256; int n = 128; int k = 64;

    // float *h_A, *h_B, *h_C;
    // // allocate memory in cpu
    // h_A = new float[m * n];
    // h_B = new float[n * k];
    // h_C = new float[m * k];

    // for (int i = 0; i < m * n; ++i) {
    //     h_A[i] = 1.0f; // Initialize matrices with some values
    //     // h_B[i] = 2.0f;
    // }
    // for (int i = 0; i < n*k; ++i) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    // allocate memory in gpu
    cudaMalloc((void**)&d_A, sizeof(float) * m * n);
    cudaMalloc((void**)&d_B, sizeof(float) * n * k);
    cudaMalloc((void**)&d_C, sizeof(float) * m * k);

    // Copy result matrix from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

    // Define grid and block dimensions for kernel execution
    dim3 dimBlock(16, 16); // 16x16 threads per block
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y); // the dimensions of the grid of thread blocks

    // Launch kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

    // std::cout<< h_C<<std::endl;
    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // delete[] h_A;
    // delete[] h_B;
    // delete[] h_C;
    // return 0;
}

