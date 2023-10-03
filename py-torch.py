import numpy as np
import example
import time
import torch

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
    h_C = h_A @ h_B

    # print("result", h_C)
    total_time += time.time() - start
    # print(h_C - h_A @ h_B )

print("use time:", total_time)
