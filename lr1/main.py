import numpy as np
import time
from CPUM import CPUM
from GPUM import GPUM


def get_cpu_time(matrix1, matrix2):
    try:
        cpum = CPUM(matrix1, matrix2)
        start_time = time.time()
        result_matrix = cpum.compute()
        end_time = time.time()
        true_result = matrix1 @ matrix2
        error = np.abs(np.sum(true_result - result_matrix))
        comput_time = end_time - start_time
        return error, comput_time
    except TypeError:
        print('Тип матриц должен быть NumPy 2D array.')

def get_gpu_time(matrix1, matrix2):
    try:
        gpum = GPUM(matrix1, matrix2)
        _ = gpum.compute() # Первый запуск = завтраты на компиляцию
        start_time = time.time()
        result_matrix = gpum.compute()
        end_time = time.time()
        true_result = matrix1 @ matrix2
        error = np.abs(np.sum(true_result - result_matrix))
        comput_time = end_time - start_time
        return error, comput_time
    except TypeError:
        print('Тип матриц должен быть NumPy 2D array.')

def print_results(er1, t1, er2, t2):
    print('Результаты вычислений:')
    print('--')
    print(f'CPU: время - {t1} сек; ошибка - {er1}')
    print('--')
    print(f'GPU: время - {t2} сек; ошибка - {er2}')
    print('--')
    print(f'GPU быстрее CPU в {t1 / t2} раз')
    print('--')

if __name__=="__main__":
    # Размерность матриц A и B
    matrix_size = 1000

    # Случайное создание матриц A и B
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    cerror, ctime = get_cpu_time(matrix_a, matrix_b) # Параллельное вычисление на центральном процессоре
    gerror, gtime = get_gpu_time(matrix_a, matrix_b) # Параллельное вычисление на графическом процессоре

    print_results(cerror, ctime, gerror, gtime)
