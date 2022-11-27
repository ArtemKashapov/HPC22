from random import getrandbits, randint
import numpy as np
from time import localtime
from pathlib import Path


SIGMA = [bytearray([i]) for i in range(256)]
OUT_FOLDER = Path('outputs/')

def get_rndm_byte_string(lngth: int) -> bytearray:
    return bytearray(getrandbits(8) for _ in range(lngth))

def generate_and_save_input_data(search_buffer_size: int, substrings_number: int, min_sbstr_sz=2, max_sbstr_sz=5) -> tuple[bytearray, list]:
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

def pre_process(search_buffer: bytearray, sub_strings: list, alphabet=SIGMA) -> tuple[dict, np.ndarray]:
    elem2pairs = dict()
    ss_sz = len(sub_strings)
    R_matrix = np.zeros((ss_sz, len(search_buffer)), dtype=np.int32)
    for n in range(ss_sz):
        R_matrix[n, :] = len(sub_strings[n])

    for i in range(256):
        sig = alphabet[i]
        cur_set = list()
        for n in range(ss_sz):
            cur_strng = sub_strings[n]
            for k in range(R_matrix[n, 0]):
                if bytearray([cur_strng[k]]) == sig:
                    cur_set.append((n, k))
        elem2pairs[str(sig)] = cur_set

    return elem2pairs, R_matrix


def main_iter(R_matrix: np.ndarray, elem2pairs:dict, search_buffer: bytearray) -> np.ndarray:
    _, sbuf_sz = R_matrix.shape
    for i in range(sbuf_sz):
        pairs = elem2pairs[str(bytearray([search_buffer[i]]))]
        if pairs:
            for pair in pairs:
                n, k = pair[0], pair[1]
                R_matrix[n, i - k] -= 1
    return R_matrix

def interp(R_matrix: np.ndarray) -> None:
    match_num = np.sum(R_matrix == 0)
    subs_sz, sbuf_sz = R_matrix.shape

    if match_num != 0:
        print(f'Совпадения найдены в количестве {match_num} штук.')
        for n in range(subs_sz):
            if R_matrix[n, :].sum() != 0:
                mathes = list()
                for j in range(sbuf_sz):
                    if R_matrix[n, j] == 0:
                        mathes.append(j)
                for j_match in mathes:
                    print(f'Подстрока {n} нашлась в буфере поиска на позиции i = {j_match}')
    else:
        print('Совпадения не найдены.')