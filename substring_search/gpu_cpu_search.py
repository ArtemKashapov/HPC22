import argparse
from time import perf_counter

import numpy as np
from numba import cuda as CUDA

from funcs import (BLOCKS_PER_GRID, THREADS_PER_BLOCK,
                   generate_and_save_input_data, interp, main_iter,
                   pre_process, pre_process4cuda, prepare_inputs4cuda)


@CUDA.jit
def run(srh_buf_arr, pairs_arr, r_matrix):
    ix = CUDA.threadIdx.x + CUDA.blockDim.x * CUDA.blockIdx.x
    threads_per_grid = CUDA.blockDim.x * CUDA.gridDim.x
    
    _, n1 = r_matrix.shape
    k0, _ = pairs_arr.shape
    for i in range(ix, n1, threads_per_grid):
        cur_elem = srh_buf_arr[i]
        for ii in range(k0):
            if cur_elem == pairs_arr[ii, 0]:
                r_matrix[pairs_arr[ii, 1], i - pairs_arr[ii, 2]] -= 1
            elif cur_elem < pairs_arr[ii, 0]:
                break

def run_on_cpu(srch_buf, sbstrs):
    tic = perf_counter()
    alph_pairs, matrix_R = pre_process(srch_buf, sbstrs)
    matrix_R = main_iter(matrix_R, alph_pairs, srch_buf)
    toc = perf_counter()
    exec_time = (toc - tic) * 1e3
    match_num = interp(matrix_R, show_pos=False)
    return matrix_R, exec_time, match_num

def run_on_gpu(srch_buf, sbstrs, n, mx):
    srch_buf_arr, _ = prepare_inputs4cuda(srch_buf, sbstrs, n, mx)
    pairs_arr, r_matrix = pre_process4cuda(srch_buf, sbstrs)

    dev_srch_buf_arr = CUDA.to_device(srch_buf_arr)
    dev_pairs_arr = CUDA.to_device(pairs_arr)
    dev_r_matrix = CUDA.to_device(r_matrix)
    # Выполнение один раз, чтобы не учитывать время компиляции
    run[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_srch_buf_arr, dev_pairs_arr, dev_r_matrix)
    _ = dev_r_matrix.copy_to_host()
    CUDA.synchronize()
    # % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    
    dev_r_matrix = CUDA.to_device(r_matrix)
    tic = perf_counter()
    pairs_arr, r_matrix = pre_process4cuda(srch_buf, sbstrs)
    run[BLOCKS_PER_GRID, THREADS_PER_BLOCK](dev_srch_buf_arr, dev_pairs_arr, dev_r_matrix)
    matrix_R = dev_r_matrix.copy_to_host()
    toc = perf_counter()
    exec_time = (toc - tic) * 1e3
    match_num = interp(matrix_R, show_pos=False)
    return matrix_R, exec_time, match_num

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--h', type=int, required=True)
    arg_parser.add_argument('--n', type=int, required=True)
    arg_parser.add_argument('--mn', type=int, required=False, default=2)
    arg_parser.add_argument('--mx', type=int, required=False, default=5)
    args = arg_parser.parse_args()

    srch_buf, sbstrs = generate_and_save_input_data(search_buffer_size=args.h, substrings_number=args.n, min_sbstr_sz=args.mn, max_sbstr_sz=args.mx)

    r_matrix_cpu, exec_time_cpu, match_num_cpu = run_on_cpu(srch_buf, sbstrs)
    r_matrix_gpu, exec_time_gpu, match_num_gpu = run_on_gpu(srch_buf, sbstrs, args.n, args.mx)
    # interp(r_matrix_gpu, show_pos=True)
    # interp(r_matrix_cpu, show_pos=True)
    print('--')
    print(f'Время выполнения на CPU: {exec_time_cpu} мс.')
    print('--')
    print(f'Время выполнения на GPU: {exec_time_gpu} мс.')
    print('--')
    print(f'Результаты совпадают: {match_num_cpu == match_num_gpu}.')


