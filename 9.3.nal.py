# -*- coding: utf-8 -*-
# stohastični populacijski modeli: model zajci-lisice

import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ProcessPriority import *
from matplotlib.colors import ListedColormap, BoundaryNorm

def populacija(Z0,L0,dt,n_max_iters=1e3,alpha=1.,beta=1.):
    Z=Z0
    L=L0
    smrt_z = 4*alpha
    smrt_l = 5*beta
    roj_z = 5*alpha
    roj_l = 4*beta
    n_iter = 0
    zajci=[Z0]
    lisice=[L0]
    gen = np.random.default_rng()
    # max_pop = 10*max([Z0,L0]) #največja možna populacija
    while n_iter<n_max_iters and Z != 0 and L != 0:
        Z = Z + gen.poisson(roj_z*Z*dt) - gen.poisson(smrt_z*Z*dt) - gen.poisson(alpha/L0*Z*L*dt)
        L = L + gen.poisson(roj_l*L*dt) - gen.poisson(smrt_l*L*dt) + gen.poisson(beta/Z0*Z*L*dt)
        n_iter+=1
        zajci.append(Z)
        lisice.append(L)
    t = np.linspace(0,n_iter,n_iter+1)*dt
    return np.array(zajci), np.array(lisice), t, n_iter
def izumrtje(data):
    j,k,Z0,L0,dt,n_max_iters,alpha,beta,n_ponovitev = data
    r=np.zeros((n_ponovitev))
    napaka=0
    pr=0
    time=[]
    for i in range(n_ponovitev):
        try:
            Z,L,t,n = populacija(Z0,L0,dt,n_max_iters,alpha,beta)
            if L[-1] == 0:
                r[i] = 1
            elif Z[-1] == 0:
                r[i] = 2
            else: r[i] = 0
            if n==n_max_iters: pr+=1
            time.append(t[-1])
        except:
            r[i]=0#dummy
            napaka+=1
            # print(f' Error in random number! Initial: Z0 = {Z0}, L0 = {L0}')
    # print(f'{napaka} / {n_ponovitev} napak v simulaciji z ZP Z0 = {Z0}, L0 = {L0}')
    # print(f'{pr} krat doseženo max št. iteracij!')
    unique, counts = np.unique(r, return_counts=True)
    index = np.argmax(counts)
    return unique[index], j,k, time

HighPriority()

if __name__ == '__main__':
    Z0=100 #ZP
    L0=50
    alpha=1. #scale factors
    beta=1.
    dt=0.1
    n_max_iters = 1e6 #ali izumrtje

    Z,L,t, n_iter = populacija(Z0,L0,dt,n_max_iters,alpha,beta)
    print(f'Izvedenih {n_iter} korakov.')
    if L[-1] == 0: print('Lisice so izumrle.')
    elif Z[-1] == 0: print('Zajci so izumrli.')

    plt.plot(L0,Z0,'ko',label='ZP')
    plt.plot(L,Z,'-')
    plt.grid()
    plt.xlabel('lisice')
    plt.ylabel('zajci')
    plt.title(f'$\Delta t = {dt}$')
    plt.savefig(f'./slike/T_L_fazni_dt{dt}.pdf')
    plt.show()

    plt.plot(t,Z,label='zajci')
    plt.plot(t,L,label='lisice')
    plt.grid()
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('Z, L')
    plt.title(f'$\Delta t = {dt}$')
    plt.savefig(f'./slike/Z_L_dt{dt}.pdf')
    plt.show()
    exit()

    #################################
    # sensitivity study ZP

    Z0_list = np.linspace(10, 100, 11)
    L0_list = np.linspace(10, 100, 11)

    alpha=1.
    beta=1.
    dt=0.001
    n_max_iters = 1e6

    # 1. zaporedno
    # t1z=timer()
    # data_zap = np.zeros((len(Z0_list), len(L0_list)))
    # for i in range(len(Z0_list)):
    #     Z0 = Z0_list[i]
    #     for j in range(len(L0_list)):
    #         L0 = L0_list[j]
    #         Z,L,t, n_iter = populacija(Z0,L0,dt,n_max_iters,alpha,beta)
    #         # print(f'Izvedenih {n_iter} korakov.')
    #         if L[-1] == 0:
    #             # print('Lisice so izumrle.')
    #             data_zap[i,j] = 1
    #         elif Z[-1] == 0:
    #             # print('Zajci so izumrli.')
    #             data_zap[i,j] = 2
    #         else:
    #             data_zap[i,j] = 0
    #             # print(f'Preostala populacija {Z[-1]} zajcev in {L[-1]} lisic.')
    # t2z=timer()
    # print(f't_zap = {t2z-t1z} s')

    # 2. vzporedno
    # t1=timer()
    # n_ponovitev = 20
    # max_workers = 18
    # data = np.zeros((len(Z0_list), len(L0_list)))

    # # Create a list of tasks
    # tasks = [(j,k,Z0_list[j],L0_list[k],dt,n_max_iters,alpha,beta,n_ponovitev) for j in range(len(Z0_list)) for k in range(len(L0_list))]

    # # Run tasks in parallel
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     # Initialize the progress bar
    #     progress_bar = tqdm(total=len(tasks), desc='Processing')

    #     # Function to update the progress bar
    #     def update_progress(*args):
    #         progress_bar.update()

    #     # Create a list of futures
    #     futures = [executor.submit(izumrtje, task) for task in tasks]
    #     for future in futures:
    #         future.add_done_callback(update_progress)

    #     # Wait for all futures to complete and collect results
    #     results = [future.result() for future in futures]

    #     # Close the progress bar
    #     progress_bar.close()

    # # Store results
    # for val, j, k, t in results:
    #     data[j, k] = val
    # t2=timer()
    # print(f't_sim = {t2-t1} s')

    # # Create the colormap
    # colors = ['black', 'orange', 'blue']
    # cmap = ListedColormap(colors)

    # # Define the boundaries
    # bounds = [-0.5, 0.5, 1.5, 2.5]
    # norm = BoundaryNorm(bounds, cmap.N)

    # plt.imshow(data, cmap=cmap, norm=norm)
    # plt.xlabel('zajci')
    # plt.ylabel('lisice')
    # cbar=plt.colorbar(ticks=[0, 1, 2],label='izumrtje populacije')
    # cbar.ax.set_yticklabels(['nedoločeno', 'lisic', 'zajcev'])
    # # plt.savefig(f'./slike/Z_L_colormap_dt{dt}.pdf')
    # plt.show()

    #################################
    # čas izumrtja
    Z0=200
    L0=50
    alpha=1.
    beta=1.
    dt=0.001
    n_max_iters = 1e6

    t1=timer()
    n_ponovitev = 10000 #željeno
    max_workers = 18
    n_ponovitev_proc = n_ponovitev//max_workers #razdelitev na procese!

    # Create a list of tasks
    k=1 #dummy
    tasks = [(i,k,Z0,L0,dt,n_max_iters,alpha,beta,n_ponovitev_proc) for i in range(max_workers)]

    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Initialize the progress bar
        progress_bar = tqdm(total=len(tasks), desc='Processing')

        # Function to update the progress bar
        def update_progress(*args):
            progress_bar.update()

        # Create a list of futures
        futures = [executor.submit(izumrtje, task) for task in tasks]
        for future in futures:
            future.add_done_callback(update_progress)

        # Wait for all futures to complete and collect results
        results = [future.result() for future in futures]

        # Close the progress bar
        progress_bar.close()

    # Store results
    time=[]
    for val, j, k, t in results:
        time.extend(t)
    t2=timer()
    print(f't_sim = {t2-t1} s')

    fig,ax=plt.subplots()
    plt.hist(time, bins=100, edgecolor='k', fill=False, histtype='step')
    avg_1=np.average(time)
    plt.axvline(avg_1, c='k', ls='--')
    plt.text(avg_1/max(time)+0.03, 1, f'$t_{{avg}}={avg_1:.2f}$', c='k', horizontalalignment='left', verticalalignment='top', rotation='vertical',transform = ax.transAxes)
    plt.xlabel('$t$')
    plt.ylabel('pogostost')
    plt.grid()
    # plt.savefig('./slike/izumrtje_Z_L.pdf')
    plt.show()