from time import perf_counter
from funcs import sum_task, get_vector, VEC_SIZE


vec_size = VEC_SIZE
vec = get_vector(vec_size=vec_size)

tic = perf_counter()
total_sum = sum_task(vec=vec, start=0, end=vec_size)
toc = perf_counter()

print(f'Vector size: {vec_size},') 
print(f'Elapsed time is {toc - tic} sec')