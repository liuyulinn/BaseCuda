import numpy as np
import time

m, n, k = 256, 128, 64
h_A = np.ones((m, n), dtype=np.float32)
h_B = np.ones((n, k), dtype=np.float32) * 2.0
h_C = np.zeros((m, k), dtype=np.float32)

start = time.time()
h_C = h_A @ h_B

print("use time:", time.time() - start)
print("result", h_C)
