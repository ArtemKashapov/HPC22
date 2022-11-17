from CPUV import CPUV
import numpy as np
from time import perf_counter

def compute_simple(vec):
    s = 0.0
    for i in range(len(vec)):
        s += vec[i]
    return s

def get_vector(n_sz):
    vec = np.arange(vsz, dtype=np.float32)
    vec /= vec.sum()
    return vec

if __name__=="__main__":
    vsz = 10000000
    vec = get_vector(vsz)

    num_trys = 51
    timing = np.empty(num_trys)
    for j in range(num_trys):
        tic = perf_counter()
        s = compute_simple(vec)
        toc = perf_counter()
        assert s, 1
        timing[j] = (toc - tic)*1e3
    print(f'Один поток: Время расчета ({timing.mean():.0f} ± {timing.std():.0f}) мс')

    cpuv = CPUV(vec=vec, chunksz=1000000, pool_sz=2)
    timing = np.empty(num_trys)
    for j in range(num_trys):
        tic = perf_counter()
        s = cpuv.compute()
        toc = perf_counter()
        assert s, 1
        timing[j] = (toc - tic)*1e3
    print(f'Много потоков: Время расчета ({timing.mean():.0f} ± {timing.std():.0f}) мс')


