from numba import cuda as CUDA
from time import perf_counter
from funcs import VEC_SIZE, get_vector, THREADS_PER_BLOCK, BLOCKS_PER_GRID, sum_task_per_block


vec_size = VEC_SIZE
vec = get_vector(vec_size)

dev_vec = CUDA.to_device(vec)
dev_part_sums = CUDA.device_array((BLOCKS_PER_GRID,), dtype=vec.dtype)

sum_task_per_block[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_vec, dev_part_sums)
total_sum = dev_part_sums.copy_to_host().sum()
CUDA.synchronize()

tic = perf_counter()
sum_task_per_block[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_vec, dev_part_sums)
total_sum = dev_part_sums.copy_to_host().sum()
CUDA.synchronize()
toc = perf_counter()

print(f'Vector size: {vec_size},')
print(f'Elapsed time is {toc - tic} sec')

# os.system('conda activate hpc')
# os.system(f'mpiexec -n {n_cpu} python lr2\\vector_multi_cpu.py')