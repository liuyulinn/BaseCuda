#include <cuda_runtime.h>
#include <cuda.h>
#include "mul.h"
#include <pybind11/numpy.h>
#include<pybind11/pybind11.h>

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

// namespace py = pybind11;
 
// __global__ void matrix_glbal_mul(float* arr_a, float* arr_b, float* res, int a_shape_0, int a_shape_1, int b_shape_1)
// {
// 	//a_shape_0，a_shape_1分别为第一个数组的行数和列数，b_shape_1为第二个数组的列数
// 	int x = threadIdx.x + blockIdx.x * blockDim.x; //   定位到res的列索引
// 	int y = threadIdx.y + blockIdx.y * blockDim.y; //   定位到res的行索引
 
// 	float Pvalue = 0;
// 	for (int k = 0; k < a_shape_1; ++k)
// 		Pvalue += arr_a[y * a_shape_1 + k] * arr_b[k * a_shape_1 + x];
 
// 	res[y * a_shape_1 + x] = Pvalue;
// }
 
 
py::array_t<float> np_multiply(py::array_t<float> &arr_a, py::array_t<float> &arr_b)
{
	//可通过此函数传入python中的numpy.ndarray数据，在C++中表现为py::array_t<T>格式。
	py::buffer_info bufA = arr_a.request(), bufB = arr_b.request();
	//request方法活得对py::array_t<T>的绑定，包括维度、数据指针、size、shape等参数
 
	const int a_shape_0 = bufA.shape[0], a_shape_1 = bufA.shape[1], b_shape_1 = bufB.shape[1];
	//分别是A的行数、列数、B的列数
	// std::cout << a_shape_0 << a_shape_1 << b_shape_1 << std::endl;
 
	auto result = py::array_t<float>(a_shape_0 * b_shape_1);
	result.resize({ a_shape_0, b_shape_1 });
 
	py::buffer_info bufResult = result.request();
	float *ptrA = (float *)bufA.ptr,
		*ptrB = (float *)bufB.ptr,
		*ptrResult = (float *)bufResult.ptr;  //获得数据指针
 
 
	float *d_a, *d_b, *d_res;
    //cudaMalloc needs a pointer to pointer as it needs to modify the original pointer d_a 
    //to point to the allocated device memory.
	cudaMalloc((void **)&d_a, a_shape_0 * a_shape_1 * sizeof(float));
	cudaMalloc((void **)&d_b, a_shape_1 * b_shape_1 * sizeof(float));
	cudaMalloc((void **)&d_res, a_shape_0 * b_shape_1 * sizeof(float));
	cudaMemcpy(d_a, ptrA, a_shape_0 * a_shape_1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, ptrB, a_shape_1 * b_shape_1 * sizeof(float), cudaMemcpyHostToDevice);
 
	
	//constexpr const int TP = 10;
	//dim3 block(TP, TP);
	//dim3 grid(a_shape_0 / TP, b_shape_1 / TP);
 
    constexpr const int TP = 16;
    dim3 block(TP, TP);
    dim3 grid((a_shape_0 + TP - 1) / TP, (b_shape_1 + TP - 1) / TP);
 
 
	matrixMultiply <<< grid, block >>> (d_a, d_b, d_res, a_shape_0, a_shape_1, b_shape_1);
	cudaMemcpy(ptrResult, d_res, a_shape_0 * b_shape_1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
 
	return result;
}
 