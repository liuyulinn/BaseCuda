#include <cuda_runtime.h>
#include <cuda.h>
#include "mul.h"


__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
			// A (row, i), A [m, n] * B (i, col) B [n, k]
			// -> C (row, col) C [m, k]
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row*k + col] = sum;
    }
}

__host__ float* npMultiply(float* arr_a, float* arr_b, int a_shape_0, int a_shape_1, int b_shape_1){ 
 
	// float *d_a, *d_b, *d_res;
	float *d_res;
	cudaMalloc((void **)&d_res, a_shape_0 * b_shape_1 * sizeof(float));
 
    constexpr const int TP = 16;
    dim3 block(TP, TP);
    dim3 grid((b_shape_1+ TP - 1) / TP, (a_shape_0 + TP - 1) / TP);
 
 
	matrixMultiply <<< grid, block >>> (arr_a, arr_b, d_res, a_shape_0, a_shape_1, b_shape_1);

 
	return d_res;
}

__host__ float* tensorMultiply(float* arr_a, float* arr_b, int a_shape_0, int a_shape_1, int b_shape_1)
{ 
	float * res;
	cudaMalloc((void **)&res, a_shape_0 * b_shape_1 * sizeof(float));

    constexpr const int TP = 16;
    dim3 block(TP, TP);
    dim3 grid((b_shape_1+ TP - 1) / TP, (a_shape_0 + TP - 1) / TP);
 
	matrixMultiply <<< grid, block >>> (arr_a, arr_b, res, a_shape_0, a_shape_1, b_shape_1);
 
	return res;
}
