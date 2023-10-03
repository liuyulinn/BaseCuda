// #include<pybind11/pybind11.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include <pybind11/numpy.h>
#include "mul.h"

torch::Tensor tensor_multiply(torch::Tensor arr_a, torch::Tensor arr_b)
{
	int a_shape_0 = arr_a.size(0), a_shape_1 = arr_a.size(1), b_shape_1 = arr_b.size(1);

	auto result = torch::from_blob(
		tensorMultiply(arr_a.data_ptr<float>(), arr_b.data_ptr<float>(), a_shape_0, a_shape_1, b_shape_1),
		torch::IntArrayRef{a_shape_0, b_shape_1},
		torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat)
	).clone();

	return result;
 
	// return result;
}
 
 
py::array_t<float> np_multiply(py::array_t<float> &arr_a, py::array_t<float> &arr_b)
{
	//可通过此函数传入python中的numpy.ndarray数据，在C++中表现为py::array_t<T>格式。
	py::buffer_info bufA = arr_a.request(), bufB = arr_b.request();
	//request方法活得对py::array_t<T>的绑定，包括维度、数据指针、size、shape等参数
 
	const int a_shape_0 = bufA.shape[0], a_shape_1 = bufA.shape[1], b_shape_1 = bufB.shape[1];
	std::cout<< a_shape_0 << " " << a_shape_1 << " " << b_shape_1 << " " << std::endl;
	//分别是A的行数、列数、B的列数
	// std::cout << a_shape_0 << a_shape_1 << b_shape_1 << std::endl;
 
	auto result = py::array_t<float>(a_shape_0 * b_shape_1);
	result.resize({ a_shape_0, b_shape_1 });
 
	py::buffer_info bufResult = result.request();
	float *ptrA = (float *)bufA.ptr,
		*ptrB = (float *)bufB.ptr,
		*ptrResult = (float *)bufResult.ptr;  //获得数据指针
 
 
	float *d_a, *d_b; // *d_res;
	cudaMalloc((void **)&d_a, a_shape_0 * a_shape_1 * sizeof(float));
	cudaMalloc((void **)&d_b, a_shape_1 * b_shape_1 * sizeof(float));
	cudaMemcpy(d_a, ptrA, a_shape_0 * a_shape_1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, ptrB, a_shape_1 * b_shape_1 * sizeof(float), cudaMemcpyHostToDevice);
 
	auto d_res = npMultiply(d_a, d_b, a_shape_0, a_shape_1, b_shape_1);
 
	// matrixMultiply <<< grid, block >>> (d_a, d_b, d_res, a_shape_0, a_shape_1, b_shape_1);
	cudaMemcpy(ptrResult, d_res, a_shape_0 * b_shape_1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	// cudaFree(d_res);
 
	return result;
}
 
 
 
 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

	m.doc() = "pybind11 example module";
	// m.def("matrix_glbal_mul", &matrix_glbal_mul, "Multuply tow arrays");
	m.def("np_multiply", &np_multiply, "Multuply tow arrays");
	m.def("tensor_multiply", &tensor_multiply, "Multuply tow arrays");
 
}