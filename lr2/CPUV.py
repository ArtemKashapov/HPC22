import numpy as np
import multiprocessing as mp
from math import ceil


class CPUV:
    def __init__(self, vec: np.ndarray, pool_sz=4, chunksz = 100) -> None:
        self.vec = vec
        self.vsz = vec.shape[0]
        self.pool_sz = pool_sz
        self.chunksz = chunksz
        self.ntasks = ceil(self.vsz / chunksz)
        self.start_end_arr = [(i * chunksz, min((i + 1) * chunksz, self.vsz)) for i in range(self.ntasks)]

    def task(self, start, end):
        return np.sum(self.vec[start:end])

    def compute(self):
        with mp.Pool(self.pool_sz) as pool:
            sum_res = pool.starmap(self.task, self.start_end_arr)
        return np.sum(sum_res)