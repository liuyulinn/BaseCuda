import numpy as np
import time

m, n, k = 256, 256, 128
total_time = 0
for i in range(100):
    # h_A = np.ones((m, n), dtype=np.float32)
    # h_B = np.ones((n, k), dtype=np.float32) * 2.0
    # h_C = np.zeros((m, k), dtype=np.float32)
    h_A = np.random.rand(m, n).astype(np.float32)
    h_B = np.random.rand(n, k).astype(np.float32) 
    h_C = np.zeros((m, k), dtype=np.float32)

    start = time.time()
    h_C = h_A @ h_B
    total_time += time.time() - start

print("use time:", total_time)
# h_A = np.ones((m, n), dtype=np.float32)
# h_B = np.ones((n, k), dtype=np.float32) * 2.0
# h_C = np.zeros((m, k), dtype=np.float32)

# start = time.time()
# h_C = h_A @ h_B

# print("use time:", time.time() - start)
# print("result", h_C)
