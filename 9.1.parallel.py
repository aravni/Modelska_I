# -*- coding: utf-8 -*-
# stohastični populacijski modeli: paralelno izvajanje

import numpy as np 
import matplotlib.pyplot as plt 
from math import floor, ceil
from timeit import default_timer as timer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ProcessPriority import *

def izumrtje(N0,beta,dt=1):
    t=0
    N=N0
    while N>0:
        p = np.random.poisson(beta*N*dt)
        N = N - p
        t += dt
    return t
def smrti_rojstva(N0,beta_s,beta_r,dt=1):
    tmax=10**4 * dt #max 10^4 iteracij
    t=0
    N=N0
    while N>0 and t<tmax:
        p = np.random.poisson(beta_r*N*dt) - np.random.poisson(beta_s*N*dt) # + rojstva - smrti
        N = N + p
        t += dt
    return t

def simulate(N0, beta, dt, n_max):
    t = []
    for i in range(n_max):
        # t_i = izumrtje(N0, beta, dt)
        t_i = smrti_rojstva(N0,1.,beta,dt) # ratio = rojstva / smrti = beta, smrti = 1
        t.append(t_i)
    return np.average(t) #np.mean(t) enako
    #return max(t)
def run_simulation(args):
    j, dt, k, beta, N0, n_max = args
    max_t = simulate(N0, beta, dt, n_max)
    return j, k, max_t

def parallel(max_workers, N0, beta_list, dt_list, n_max):
    t_list = np.zeros((len(dt_list), len(beta_list)))

    # Create a list of tasks
    tasks = [(j, dt_list[j], k, beta_list[k], N0, n_max) for j in range(len(dt_list)) for k in range(len(beta_list))]

    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Initialize the progress bar
        progress_bar = tqdm(total=len(tasks), desc='Processing')

        # Function to update the progress bar
        def update_progress(*args):
            progress_bar.update()

        # Create a list of futures
        futures = [executor.submit(run_simulation, task) for task in tasks]
        for future in futures:
            future.add_done_callback(update_progress)

        # Wait for all futures to complete and collect results
        results = [future.result() for future in futures]

        # Close the progress bar
        progress_bar.close()

    # Store results in t_list
    for j, k, max_t in results:
        t_list[j, k] = max_t

    return t_list

#########################
# odvisnost dt, beta - čas izumrtja
HighPriority()

# ProcessPool -> veliko bolje! od ThreadPool
if __name__ == '__main__':
    beta_list = [0.3,0.4,0.5,0.7,0.8]
    dt_list = [0.2,0.3]

    beta_list = np.arange(0.1,1,0.1)
    dt_list = np.arange(0.01,0.1,0.01)

    beta_list = np.linspace(0.1,1.,50)
    beta_list = np.linspace(0.1,0.9,50) #če beta -> ratio
    dt_list = np.linspace(0.01,10.,50)

    n_max = 10000 #za povprečenje OK
    N0 = 250
    max_workers = 18
    num_xticks = num_yticks = 10
    t_list = np.zeros((len(dt_list),len(beta_list)))
    print('št. izračunov: ',len(beta_list), ' x ',len(dt_list),' = ',len(beta_list)*len(dt_list)) 

    # Determine the tick positions and labels
    xtick_positions = np.linspace(0, len(beta_list) - 1, num_xticks, dtype=int)
    xticks = [f'{beta_list[i]:.2f}' for i in xtick_positions]

    ytick_positions = np.linspace(0, len(dt_list) - 1, num_yticks, dtype=int)
    yticks = [f'{dt_list[i]:.2f}' for i in ytick_positions]

    # Set the tick positions and labels for the x-axis
    plt.gca().set_xticks(xtick_positions)
    plt.gca().set_xticklabels(xticks)

    # Set the tick positions and labels for the y-axis
    plt.gca().set_yticks(ytick_positions)
    plt.gca().set_yticklabels(yticks)

    plt.title('test')
    plt.xlabel('$\\beta$')
    plt.ylabel('$dt$')
    plt.show()

    ############# 1. parallel simulation ProcessPool

    t1=timer()
    t_list = parallel(max_workers, N0, beta_list, dt_list, n_max)
    t2=timer()
    print(f'vzporedno ProcessPool t_sim = {t2-t1:.5f} s, max_workers = {max_workers}')

    im=plt.imshow(t_list,origin='lower', aspect='auto', cmap='turbo') #jet vs turbo?
    plt.colorbar(im, label='povprečen $t_{izumrtja}$ [s]')

    # Determine the tick positions and labels
    xtick_positions = np.linspace(0, len(beta_list) - 1, num_xticks, dtype=int)
    xticks = [f'{beta_list[i]:.2f}' for i in xtick_positions]

    ytick_positions = np.linspace(0, len(dt_list) - 1, num_yticks, dtype=int)
    yticks = [f'{dt_list[i]:.2f}' for i in ytick_positions]

    # Set the tick positions and labels for the x-axis
    plt.gca().set_xticks(xtick_positions)
    plt.gca().set_xticklabels(xticks)

    # Set the tick positions and labels for the y-axis
    plt.gca().set_yticks(ytick_positions)
    plt.gca().set_yticklabels(yticks)

    plt.xlabel('$\\beta$') #le smrti
    plt.xlabel('ratio') #rojstva + smrti
    plt.ylabel('$dt$')
    plt.title(f'$N_0={N0}$, povprečje {n_max} ponovitev')
    # plt.savefig(f'izumrtje_dt_beta.pdf')
    plt.show()

    ######################## ThreadPool -> malo hitreje od zaporedno, nekaj sec
    # t1=timer()
    # tasks = [(j, dt_list[j], k, beta_list[k], N0,n_max) for j in range(len(dt_list)) for k in range(len(beta_list))]
    # # Run tasks in parallel
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     progress_bar = tqdm(total=len(tasks), desc='Processing')
    #     def update_progress(*args):
    #         progress_bar.update()
    #     futures = [executor.submit(run_simulation, task) for task in tasks]
    #     for future in futures:
    #         future.add_done_callback(update_progress)
    #     results = [future.result() for future in futures]
    #     progress_bar.close()
    # for j, k, max_t in results:
    #     t_list[j, k] = max_t
    # t2=timer()
    # print(f'vzporedno ThreadPool t_sim = {t2-t1:.5f} s')

    # ######## 1. zaporedno
    # t1=timer()
    # for j in tqdm(range(len(dt_list)), desc='dt'):
    #     dt = dt_list[j]
    #     for k in tqdm(range(len(beta_list)), desc='beta', leave=False):
    #         beta = beta_list[k]
    #         t = []
    #         for i in tqdm(range(n_max), desc='i', leave=False):
    #         # for i in range(n_max):
    #             t_i = izumrtje(N0,beta,dt)
    #             t.append(t_i)
    #         t_list[j,k]=max(t)
    # t2=timer()
    # t2=timer()
    # print(f'zaporedno t_sim = {t2-t1:.5f} s')