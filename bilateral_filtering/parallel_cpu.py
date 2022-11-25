from math import ceil
from time import perf_counter

import numpy as np
from mpi4py import MPI

from Imlib import (FILE, SIG_D, SIG_R, Filter, arr2img, open_image,
                   parallel_task)

comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()
p = comm.Get_size()

tic = perf_counter()

pad_img_arr, fltr = None, None
if current_rank == 0:
    img_arr = open_image(FILE)
    fltr = Filter(img_arr, sig_r=SIG_R, sig_d=SIG_D)
    pad_img_arr = fltr.pad()

pad_img_arr = comm.bcast(pad_img_arr)
fltr = comm.bcast(fltr)

portion = ceil(fltr.m / p)
local_start_ind = current_rank * portion
local_end_ind = min((current_rank + 1) * portion, fltr.m)
local_part = parallel_task(i_start=local_start_ind, i_end=local_end_ind, pad_img_arr=pad_img_arr, fltr_obj=fltr)

if current_rank == 0:
    fltrd_img_arr = np.zeros_like(img_arr)
    start_ind = current_rank * portion
    end_ind = min((current_rank + 1) * portion, fltr.m)
    fltrd_img_arr[start_ind:end_ind, :] = local_part
    for k in range(1, p):
        start_ind = k * portion
        end_ind = min((k + 1) * portion, fltr.m)
        fltrd_img_arr[start_ind:end_ind, :] = comm.recv(source=k)
    
    toc = perf_counter()
    arr2img(fltrd_img_arr, True)
    print(f'Pocessesors number: {p}, ')
    print(f'Elapsed time is {toc - tic} sec')
else:
    comm.send(local_part, dest=0)

MPI.Finalize

