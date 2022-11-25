import warnings
from time import perf_counter

import numpy as np
from numba import cuda as CUDA
from numba.core.errors import NumbaPerformanceWarning

from Imlib import (BLOCKS_PER_GRID, FILE, SIG_D, SIG_R, THREADS_PER_BLOCK,
                   Filter, arr2img, fliter_task, open_image)

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


img_arr = open_image(FILE)
pad_img_arr = Filter(img=img_arr, sig_d=SIG_D, sig_r=SIG_R).pad()
pad_img_arr_gpu = CUDA.to_device(pad_img_arr)
fltd_arr_gpu = CUDA.device_array_like(np.zeros_like(img_arr))

fliter_task[BLOCKS_PER_GRID, THREADS_PER_BLOCK](pad_img_arr, SIG_R, SIG_D, fltd_arr_gpu)
fltd_arr = fltd_arr_gpu.copy_to_host()
CUDA.synchronize()

tic = perf_counter()
fliter_task[BLOCKS_PER_GRID, THREADS_PER_BLOCK](pad_img_arr, SIG_R, SIG_D, fltd_arr_gpu)
fltd_arr = fltd_arr_gpu.copy_to_host()
CUDA.synchronize()
toc = perf_counter()
print(f'Elapsed time is {toc - tic} sec')

arr2img(fltd_arr, True)