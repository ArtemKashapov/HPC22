import numpy as np, numba as nb
from numba import cuda as CUDA
from cuda.cuda import CUdevice_attribute, cuDeviceGetAttribute, cuDeviceGetName, cuInit

ARRAY_DTYPE = np.float64
VEC_SIZE = 1000

THREADS_PER_BLOCK = 1024
BLOCKS_PER_GRID = 256

def get_vector(vec_size: int) -> np.ndarray:
    vec = np.arange(vec_size, dtype=ARRAY_DTYPE)
    vec /= vec.sum()
    return vec

def sum_task(vec: np.ndarray, start: int, end: int) -> ARRAY_DTYPE:
    s = 0.0
    for i in range(start, end):
        s += vec[i]
    return s

@CUDA.jit
def sum_task_per_block(vector, part_sums):
    start_ind = CUDA.threadIdx.x + CUDA.blockDim.x * CUDA.blockIdx.x
    threads_per_grid = CUDA.blockDim.x * CUDA.gridDim.x
    sum_thread = 0.0
    for i_arr in range(start_ind, vector.size, threads_per_grid):
        sum_thread += vector[i_arr]

    sum_block = CUDA.shared.array((THREADS_PER_BLOCK,), nb.float64)
    
    thread_ind = CUDA.threadIdx.x
    sum_block[thread_ind] = sum_thread

    CUDA.syncthreads()

    if thread_ind == 0:
        for i in range(1, THREADS_PER_BLOCK):
            sum_block[0] += sum_block[i]
        part_sums[CUDA.blockIdx.x] = sum_block[0]

def print_gpu_info():
    # Initialize CUDA Driver API
    (err,) = cuInit(0)

    # Get attributes
    err, DEVICE_NAME = cuDeviceGetName(128, 0)
    DEVICE_NAME = DEVICE_NAME.decode("ascii").replace("\x00", "")

    err, MAX_THREADS_PER_BLOCK = cuDeviceGetAttribute(
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0
    )
    err, MAX_BLOCK_DIM_X = cuDeviceGetAttribute(
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0
    )
    err, MAX_GRID_DIM_X = cuDeviceGetAttribute(
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0
    )
    err, SMs = cuDeviceGetAttribute(
        CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
    )

    print(f"Device Name: {DEVICE_NAME}")
    print(f"Maximum number of multiprocessors: {SMs}")
    print(f"Maximum number of threads per block: {MAX_THREADS_PER_BLOCK:10}")
    print(f"Maximum number of blocks per grid:   {MAX_BLOCK_DIM_X:10}")
    print(f"Maximum number of threads per grid:  {MAX_GRID_DIM_X:10}")