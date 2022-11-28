from random import getrandbits, randint
import numpy as np
from time import localtime
from pathlib import Path


THREADS_PER_BLOCK = 1024
BLOCKS_PER_GRID = 256

SIGMA = [i for i in range(256)]
OUT_FOLDER = Path('outputs/')

def get_rndm_byte_string(lngth: int) -> list:
    return [getrandbits(8) for _ in range(lngth)]

def generate_and_save_input_data(search_buffer_size: int, substrings_number: int, min_sbstr_sz=2, max_sbstr_sz=5) -> tuple[list, list]:
    search_buffer = get_rndm_byte_string(search_buffer_size)
    substrings = [get_rndm_byte_string(randint(min_sbstr_sz, max_sbstr_sz)) for _ in range(substrings_number)]
    exec_moment = localtime()
    f_name = str(exec_moment.tm_mday) + '_' + str(exec_moment.tm_mon) + '_' + str(exec_moment.tm_hour) + '_' + str(exec_moment.tm_min) + '_' + str(exec_moment.tm_sec) + '.txt'
    with open(OUT_FOLDER / f_name, 'w', encoding='utf-8') as file:
        file.write('Буфер поиска:\n', )
        file.write(str(search_buffer))
        file.write('\n--\n')
        file.write('Подстроки для поиска:\n')
        file.write(str(substrings))
        file.write('\n--\n')
    return search_buffer, substrings

def pre_process(search_buffer: list, sub_strings: list, alphabet=SIGMA) -> tuple[dict, np.ndarray]:
    elem2pairs = dict()
    ss_sz = len(sub_strings)
    R_matrix = np.zeros((ss_sz, len(search_buffer)), dtype=np.int64)
    for n in range(ss_sz):
        R_matrix[n, :] = len(sub_strings[n])

    for i in range(256):
        sig = alphabet[i]
        cur_set = list()
        for n in range(ss_sz):
            cur_strng = sub_strings[n]
            for k in range(R_matrix[n, 0]):
                if cur_strng[k] == sig:
                    cur_set.append((n, k))
        elem2pairs[str(sig)] = cur_set

    return elem2pairs, R_matrix

def main_iter(R_matrix: np.ndarray, elem2pairs:dict, search_buffer: list) -> np.ndarray:
    _, sbuf_sz = R_matrix.shape
    for i in range(sbuf_sz):
        pairs = elem2pairs[str(search_buffer[i])]
        if pairs:
            for pair in pairs:
                n, k = pair[0], pair[1]
                R_matrix[n, i - k] -= 1
                
    return R_matrix

def interp(R_matrix: np.ndarray, show_pos: bool) -> None:
    match_num = np.sum(R_matrix == 0)
    if match_num != 0:
        if show_pos:
            print(f'Совпадения найдены в количестве {match_num} штук.')
            for n in np.where(np.sum(R_matrix == 0, axis=1) != 0)[0]:
                for j in np.where(R_matrix[n, :] == 0)[0]:
                    print(f'Подстрока {n} нашлась в буфере поиска на позиции i = {j}')
        else:
            # print(f'Совпадения найдены в количестве {match_num} штук.')
            return match_num
    else:
        print('Совпадения не найдены.')


def pre_process4cuda(search_buffer: list, sub_strings: list) -> tuple[np.ndarray, np.ndarray]:
    elem2pairs = list()
    ss_sz = len(sub_strings)
    R_matrix = np.zeros((ss_sz, len(search_buffer)), dtype=np.int64)
    for n in range(ss_sz):
        R_matrix[n, :] = len(sub_strings[n])

    for sig in range(256):
        for n in range(ss_sz):
            cur_strng = sub_strings[n]
            for k in range(R_matrix[n, 0]):
                if cur_strng[k] == sig:
                    elem2pairs.append([sig, n, k])

    return np.array([np.array(pair, dtype=np.int64) for pair in elem2pairs], dtype=np.int64), R_matrix

def prepare_inputs4cuda(srch_buf: list, sbstrs:list, n: int, mx: int) -> tuple[np.ndarray, np.ndarray]:
    srch_buf_arr = np.array(srch_buf)
    sbstrs_arr = -1 * np.ones((n, mx), dtype=np.int64)
    for i, strng in enumerate(sbstrs):
        sbstrs_arr[i, :len(strng)] = np.array(strng)
    
    return srch_buf_arr, sbstrs_arr
