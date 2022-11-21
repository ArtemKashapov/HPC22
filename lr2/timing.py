from time import perf_counter, sleep
from funcs import sum_task, get_vector, VEC_SIZE, sum_task_per_block, BLOCKS_PER_GRID, THREADS_PER_BLOCK
import numpy as np
import subprocess
from numba import cuda as CUDA


# # # # # # # # # # MULTI GPU # # # # # # # # # #
 
vec_size = VEC_SIZE
vec = get_vector(vec_size=vec_size)

dev_vec = CUDA.to_device(vec)
dev_part_sums = CUDA.device_array((BLOCKS_PER_GRID,), dtype=vec.dtype)

sum_task_per_block[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_vec, dev_part_sums)
total_sum = dev_part_sums.copy_to_host().sum()
CUDA.synchronize()


n_starts = 101
timing = np.empty(n_starts)
for i in range(n_starts):
    tic = perf_counter()
    sum_task_per_block[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_vec, dev_part_sums)
    total_sum = dev_part_sums.copy_to_host().sum()
    CUDA.synchronize()
    toc = perf_counter()
    timing[i] = toc - tic

timing *= 1e3
print(f'At vector size {vec_size}:')
print(f'Elapsed time is ({timing.mean()} ± {timing.std()}) ms')

# # # # # # # # # # MULTI CPU # # # # # # # # # #

# vec_size = VEC_SIZE
# n_starts = 101
# timing = np.empty(n_starts)
# for i in range(n_starts):
#     output = subprocess.Popen(f'mpiexec -n {12} python lr2\\vector_multi_cpu.py', stdout=subprocess.PIPE).communicate()[0]
#     t_exec = float(output.split()[9])
#     timing[i] = t_exec
#     sleep(2)

# timing *= 1e3
# print(f'At vector size {VEC_SIZE}:')
# print(f'Elapsed time is ({timing.mean()} ± {timing.std()}) ms')

# # # # # # # # # # SINGLE CPU # # # # # # # # # #
 
# vec_size = VEC_SIZE
# vec = get_vector(vec_size=vec_size)

# n_starts = 101
# timing = np.empty(n_starts)
# for i in range(n_starts):
#     tic = perf_counter()
#     total_sum = sum_task(vec=vec, start=0, end=vec_size)
#     toc = perf_counter()
#     timing[i] = toc - tic

# timing *= 1e3
# print(f'At vector size {vec_size}:')
# print(f'Elapsed time is ({timing.mean()} ± {timing.std()}) ms')