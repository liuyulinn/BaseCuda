#include <cuda_runtime.h>
#include <cuda.h>
#include <pybind11/numpy.h>
#include<pybind11/pybind11.h>

namespace py = pybind11;
__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k);
py::array_t<float> np_multiply(py::array_t<float> &arr_a, py::array_t<float> &arr_b);
