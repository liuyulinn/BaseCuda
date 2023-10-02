import numpy as np
import example
import time
 
all_attributes = dir(example)
functions = [attr for attr in all_attributes if callable(getattr(example, attr))]

# Print the list of functions
for function_name in functions:
    print(function_name)
print("Over!")
 
m, n, k = 256, 128, 64
h_A = np.ones((m, n), dtype=np.float32)
h_B = np.ones((n, k), dtype=np.float32) * 2.0
h_C = np.zeros((m, k), dtype=np.float32)

start = time.time()
h_C = example.np_multiply(h_A, h_B)

print("use time:", time.time() - start)
print("result", h_C)
