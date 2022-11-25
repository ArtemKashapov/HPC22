from time import perf_counter

from Imlib import FILE, SIG_D, SIG_R, Filter, arr2img, open_image


if __name__=="__main__":
    tic = perf_counter()
    filtd_img = Filter(open_image(FILE), sig_r=SIG_R, sig_d=SIG_D).filt()
    toc = perf_counter()
    print(f'Pocessesors number: 1, ')
    print(f'Elapsed time is {toc - tic} sec')
    arr2img(filtd_img, True)
