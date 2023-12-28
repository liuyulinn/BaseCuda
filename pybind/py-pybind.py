import numpy as np
import example # multiple package name
                # package name specified in setup.py
import time
import torch
 
# all_attributes = dir(example)
# functions = [attr for attr in all_attributes if callable(getattr(example, attr))]

# Print the list of functions
# for function_name in functions:
#     print(function_name)
# print("Over!")
 
m, n, k = 256, 256, 128

total_time = 0
for i in range(100):
    h_A = torch.rand((m,n), dtype = torch.float32).cuda()
    h_B = torch.rand((n, k), dtype = torch.float32).cuda()
    h_C = torch.zeros((m, n), dtype = torch.float32).cuda()
    # h_A = torch.from_numpy(np.ones((m, n), dtype=np.float32)).cuda()
    # h_B = torch.from_numpy(np.ones((n, k), dtype=np.float32) * 2.0).cuda()
    # h_C = torch.from_numpy(np.zeros((m, k), dtype=np.float32)).cuda()

    start = time.time()
    h_C = example.tensor_multiply(h_A, h_B)

    # print("result", h_C)
    total_time += time.time() - start
    # print(h_C - h_A @ h_B )

print("use time:", total_time)

# # print(h_C.shape)


# h_A = np.ones((m, n), dtype=np.float32)
# h_B = np.ones((n, k), dtype=np.float32) * 2.0
# h_C = np.zeros((m, k), dtype=np.float32)

# start = time.time()
# h_C = example.np_multiply(h_A, h_B)

# print("result", h_C)
# print("use time:", time.time() - start)
# print(h_C.shape)