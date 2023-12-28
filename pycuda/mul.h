#include <cuda_runtime.h>
#include <cuda.h>
#include <pybind11/numpy.h>
#include<pybind11/pybind11.h>
#include<torch/extension.h>

namespace py = pybind11;
__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k);
// py::array_t<float> np_multiply(py::array_t<float> &arr_a, py::array_t<float> &arr_b);
__host__ float* npMultiply(float* arr_a, float* arr_b, int a_shape_0, int a_shape_1, int b_shape_1);
__host__ float* tensorMultiply(float* arr_a, float* arr_b, int a_shape_0, int a_shape_1, int b_shape_1);
