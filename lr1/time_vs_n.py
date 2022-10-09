import numpy as np
import time
from main import get_cpu_time, get_gpu_time


def simple_method(matrix1, matrix2):
    m1, n1 = matrix1.shape
 
    result_matrix = np.zeros_like(matrix1)
    for i in range(m1):
        for j in range(n1):
            result_matrix[i, j] = matrix1[i, :].dot(matrix2[:, j])
    
    return result_matrix

if __name__=="__main__":
    # Вектор размерностей матриц A и B
    matrix_size_arr = [100, 200, 400, 800, 1000, 1500, 2000]
    cpu_time_arr = list()
    gpu_time_arr = list()
    np_time_arr = list()
    for size in matrix_size_arr:
        matrix_a = np.random.rand(size, size)
        matrix_b = np.random.rand(size, size)
        cpu_time_sum = 0
        gpu_time_sum = 0
        np_time_sum = 0

        iter_num = 5
        for _ in range(iter_num):
            _, ctime = get_cpu_time(matrix_a, matrix_b) # Параллельное вычисление на центральном процессоре
            cpu_time_sum += ctime
            _, gtime = get_gpu_time(matrix_a, matrix_b) # Параллельное вычисление на графическом процессоре
            gpu_time_sum += gtime
            startt = time.time()
            _ = simple_method(matrix_a, matrix_b)
            endt = time.time()
            np_time_sum += endt - startt
        cpu_time_arr.append(cpu_time_sum / iter_num)
        gpu_time_arr.append(gpu_time_sum / iter_num)
        np_time_arr.append(np_time_sum / iter_num)
    
    print('Вектор с размерностями:', matrix_size_arr)
    print('--')
    print('CPU:', cpu_time_arr)
    print('--')
    print('GPU:', gpu_time_arr)
    print('--')
    print('Single CPU:', np_time_arr)
    print('--')
    

