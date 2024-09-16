import numpy as np
import matplotlib.pyplot as plt

def pi(N_max, case_random=True):
    n = 0 #št zadetkov
    N = 0 #št klicev
    if case_random:
        while N < N_max:
                x = np.random.random()
                y = np.random.random()
                r = x*x + y*y #kvadrat
                if r <= 1: #znotraj kroga
                    n += 1
                    N += 1
                else:
                    N += 1
    else:
        xx = np.linspace(0, 1 , int(np.sqrt(N_max)))
        yy = np.linspace(0, 1 , int(np.sqrt(N_max)))
        for x in xx:
            for y in yy:
                r = x*x + y*y #kvadrat
                if r <= 1: #znotraj kroga
                    n += 1
                    N += 1
                else:
                    N += 1
    pi = 4 * n/N
    napaka = pi - np.pi
    print(f'pi = {pi}, napaka = {napaka}')
    print(f'število točk: {N_max}')
    return pi, napaka
pi(10000,False)
