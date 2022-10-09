import numpy as np
from numba import jit

import warnings
warnings.filterwarnings("ignore")


class GPUM:
    def __init__(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.check_input()
        self.n = matrix_a.shape[0]

    def check_input(self):
        m, n = self.matrix_a.shape
        k, l = self.matrix_b.shape
        
        if (m != n) or (k != l) or (m != k):
            raise ValueError
        elif not isinstance(self.matrix_a, np.ndarray) or not isinstance(self.matrix_b, np.ndarray):
            raise TypeError        

    @jit(target_backend='cuda')
    def compute(self) -> np.ndarray:
        matrix_c = np.zeros_like(self.matrix_a)
        for i in range(self.n):
            matrix_c[i, :] = self.matrix_a[i, :].dot(self.matrix_b)
        return matrix_c

