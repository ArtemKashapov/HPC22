from math import ceil, exp
from pathlib import Path

import numpy as np
from numba import cuda as CUDA
from PIL import Image

SRC_FOLDER = Path('source/')
# FILE = 'grey_lady.bmp'
# FILE = 'grey_balls.bmp'
FILE = 'grey_pens.bmp'
SIG_R = 200
SIG_D = 200
THREADS_PER_BLOCK = (16, 16)
BLOCKS_PER_GRID = (16, 16)

def open_image(path: str) -> np.ndarray:
    img_path = SRC_FOLDER / path
    return np.array(Image.open(img_path), dtype=np.float32)

def arr2img(img_arr: np.ndarray, show_flag: bool) -> Image:
    if show_flag:
        Image.fromarray(np.uint8(img_arr)).show()
    else:
        return Image.fromarray(np.uint8(img_arr))

def parallel_task(i_start: int, i_end: int, pad_img_arr: np.ndarray, fltr_obj: object) -> np.ndarray:
    arr_part = np.empty((i_end - i_start, fltr_obj.n))
    for i in range(i_start, i_end):
        for j in range(fltr_obj.n):
            arr_part[i-i_start, j] = fltr_obj.job(i+1, j+1, pad_img_arr)
    
    return arr_part

@CUDA.jit
def fliter_task(inp, sig_r, sig_d, out):
    ix, iy = CUDA.grid(2)
    threads_per_grid_x, threads_per_grid_y = CUDA.gridsize(2)
    
    n0, n1 = inp.shape
    c0, c1 = ceil(n0 / 2 - 1), ceil(n1 / 2 - 1)
    for i0 in range(iy+1, n0-1, threads_per_grid_y):
        for i1 in range(ix+1, n1-1, threads_per_grid_x):
            k = 0.0
            s = 0.0
            for j0 in range(-1, 2):
                for j1 in range(-1, 2):
                    core = exp(pow((inp[i0+j0, i1+j1] - inp[i0, i1]) / sig_r, 2) - (pow(i0+j0-c0, 2) + pow(i1+j1-c1, 2)) / pow(sig_d, 2))
                    k += core
                    s += core * inp[i0+j0, i1+j1]
            out[i0-1, i1-1] = s / k

class Filter:
    def __init__(self, img: np.ndarray, sig_r: np.float32, sig_d: np.float32) -> None:
        self.sig_r = sig_r
        self.sig_d = sig_d
        self.img = img
        self.m, self.n = img.shape
        self.center = (self.m / 2, self.n / 2)
        self.tmp_ind = np.tile(np.linspace(-1, 1, num=3, dtype=np.int32), (3, 1))
    
    def pad(self) -> np.ndarray:
        pad_img = self.img
        pad_img = np.c_[np.zeros((self.m, 1)), pad_img]
        pad_img = np.c_[pad_img, np.zeros((self.m, 1))]
        pad_img = np.r_[np.zeros((1, self.n + 2)), pad_img]
        pad_img = np.r_[pad_img, np.zeros((1, self.n + 2))]
        return pad_img
    
    def job(self, i: np.int32, j: np.int32, pad_img: np.ndarray):
        r = np.exp((pad_img[i + self.tmp_ind, j + self.tmp_ind.T] - pad_img[i, j]) ** 2 / self.sig_r**2)
        g = np.exp((-((i + self.tmp_ind - self.center[0])**2 + (j + self.tmp_ind.T - self.center[1])**2) / self.sig_d**2))
        f = pad_img[i + self.tmp_ind, j + self.tmp_ind.T]
        return (f * g * r).sum() / (g * r).sum()

    def filt(self) -> np.ndarray:
        pad_img = self.pad()
        filt_img = np.zeros_like(self.img)
        
        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                filt_img[i-1][j-1] = self.job(i, j, pad_img)

        return filt_img