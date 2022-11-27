import argparse
from time import perf_counter

from funcs import generate_and_save_input_data, interp, main_iter, pre_process


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--h', type=int, required=True)
    arg_parser.add_argument('--n', type=int, required=True)
    arg_parser.add_argument('--mn', type=int, required=False, default=2)
    arg_parser.add_argument('--mx', type=int, required=False, default=5)
    args = arg_parser.parse_args()

    srch_buf, sbstrs = generate_and_save_input_data(search_buffer_size=args.h, substrings_number=args.n, min_sbstr_sz=args.mn, max_sbstr_sz=args.mx)

    tic = perf_counter()
    alph_pairs, matrix_R = pre_process(srch_buf, sbstrs)
    matrix_R = main_iter(matrix_R, alph_pairs, srch_buf)
    toc = perf_counter()
    interp(matrix_R)

    exec_time = (toc - tic) * 1e3
    print(f'Время исполнения: {exec_time} мс.')


