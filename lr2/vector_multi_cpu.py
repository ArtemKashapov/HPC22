from mpi4py import MPI
import numpy as np
from time import perf_counter
from math import ceil
from funcs import sum_task, get_vector, VEC_SIZE


tic = perf_counter()
comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()
p = comm.Get_size()

vec_size = VEC_SIZE
vec = get_vector(vec_size=vec_size)

total_sum = -1.0
arr_portion = ceil(vec_size / p)

local_start_ind = current_rank * arr_portion
local_end_ind = min((current_rank + 1) * arr_portion, vec_size)

part_sum = sum_task(vec=vec, start=local_start_ind, end=local_end_ind)
total_sum = comm.reduce(part_sum)

if current_rank == 0:
    toc = perf_counter()

    print(f'Vector size: {vec_size},')
    print(f'Pocesses number: {p}, ')
    print(f'Elapsed time is {toc - tic} sec')

MPI.Finalize