import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

# Load the compiled CUDA code
from pycuda.compiler import SourceModule
# mod = SourceModule("""
# __global__ void Mul(float * A, float * B, float * C,
#                    int A_shape_0,int A_shape_1,int B_shape_1) {
#     float cValue = 0;
#     int Row = blockIdx.y * blockDim.y + threadIdx.y;
#     int Col = blockIdx.x * blockDim.x + threadIdx.x;
#     if ((Row < A_shape_0) && (Col < B_shape_1)) {
#         for (int k = 0; k < A_shape_1; k++) {
#             cValue += A[Row*A_shape_1 + k] * B[k*B_shape_1 + Col];
#         }
#         C[Row*B_shape_1 + Col] = cValue;
#     }
# }
# """)

mod = SourceModule(open("mul.cu").read())

# Get a reference to the multiplyMatrix function
# print(mod.module)
Mul = mod.get_function("Mul")

m, n, k = 256, 128, 64

total_time = 0
for i in range(100):
    # h_A = np.ones((m, n), dtype=np.float32)
    # h_B = np.ones((n, k), dtype=np.float32) * 2.0
    # h_C = np.zeros((m, k), dtype=np.float32)
    h_A = np.random.rand(m, n).astype(np.float32)
    h_B = np.random.rand(n, k).astype(np.float32) 
    h_C = np.zeros((m, k), dtype=np.float32)

    # Call the CUDA function
    # If I don't manually move h_A, h_B from CPU to GPU, or move h_C from GPU to CPU,
    # I can call cuda.In(h_A) and cuda.Out(h_C)

    BLOCK_SIZE = 16
    grid=((k + BLOCK_SIZE) // BLOCK_SIZE, (m + BLOCK_SIZE) // BLOCK_SIZE, 1)

    start = time.time()
    Mul(
        cuda.In(h_A), cuda.In(h_B), cuda.Out(h_C),
        np.int32(m), np.int32(n), np.int32(k),
        block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid = grid
    )
    total_time += time.time() - start

    # print(h_A @ h_B - h_C)

# print("result", h_C)
print("use time:", total_time)



'''
warning: still can't work I don't know why
# for name, item in mod.__dict__.items():
#     # Check if the item is a function
#     if callable(item):
#         print(f"Function Name: {name}")
# Multrix = mod.get_function("Multrix")  # Use "Multrix" here, which matches the function name.

# Multrix(
#     h_A, h_B, h_C,
#     np.int32(m), np.int32(n), np.int32(k)
# )

# # Your result is now in h_C
# print(h_C)
'''