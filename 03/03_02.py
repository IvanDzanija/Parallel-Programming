import numpy as np
import time
import numba
from numba import guvectorize, vectorize, float64, float32, int32

N = 1 << 24
print(numba.get_num_threads())


# Pure python
def single_core_pi():
    acc = 0.0
    for i in range(N):
        x = (i - 0.5) / N
        acc += 1 / (1 + x * x)
    return acc * 4 / N


start_time = time.time()
ans = single_core_pi()
print(f"Single Core -> Pi: {ans} | Time : {time.time() - start_time:.5f} s")


# Numba vectorize
@vectorize([float32(int32, float32)], target="parallel")
def pi_vectorize(i, N):
    x = (i + 0.5) / N
    return 4.0 / (1.0 + x * x)


indices = np.arange(N, dtype=np.int32)


_ = pi_vectorize(indices, np.float32(N))
start_time = time.time()
results = pi_vectorize(indices, np.float32(N))
ans = np.sum(results) / N
print(f"Vectorize -> Pi: {ans} | Time : {time.time() - start_time:.5f} s")


# Numba guvectorize
@guvectorize([(int32[:], float32, float32[:])], "(m),()->()", target="parallel")
def pi_guvectorize(local_indices, N, result):
    local = 0.0
    for i in range(local_indices.shape[0]):
        x = (local_indices[i] + 0.5) / N
        local += 4.0 / (1.0 + x * x)
    result[0] = local


N = 1 << 24
indices = np.arange(N, dtype=np.int32)

for block_size in [1024, 2048, 4096]:
    num_blocks = N // block_size
    blocks = indices.reshape(num_blocks, block_size)

    start_time = time.time()
    partial = pi_guvectorize(blocks, np.float32(N))
    ans = np.sum(partial) / N
    print(
        f"GUVectorize -> Block size: {block_size} | Pi: {ans} | Time: {time.time() - start_time:.5f} s"
    )
