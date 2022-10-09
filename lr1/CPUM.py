import numpy as np
import multiprocessing as mp


class CPUM:
    def __init__(self, matrix_a: np.ndarray, matrix_b: np.ndarray, pool_sz=4) -> None:
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.check_input()
        self.n = matrix_a.shape[0]
        self.pool_sz = pool_sz

    def check_input(self):
        m, n = self.matrix_a.shape
        k, l = self.matrix_b.shape
        
        if (m != n) or (k != l) or (m != k):
            raise ValueError
        elif not isinstance(self.matrix_a, np.ndarray) or not isinstance(self.matrix_b, np.ndarray):
            raise TypeError

    def task(self, index):
        return self.matrix_a[index, :].dot(self.matrix_b)

    def compute(self):
        with mp.Pool(self.pool_sz) as pool:
            matrix_c = pool.map(self.task, list(range(self.n)))
        return np.array(matrix_c)