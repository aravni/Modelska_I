# -*- coding: utf-8 -*-
# stohastični populacijski modeli: statistika časov izumrtja, potek populacije

import numpy as np 
import matplotlib.pyplot as plt 
from math import floor, ceil
from timeit import default_timer as timer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from ProcessPriority import *
from scipy.integrate import odeint, solve_ivp
from matplotlib.colors import LogNorm
from scipy.special import factorial

def smrti(N0,beta,dt=1):
    t=0
    N=N0
    pop = [N0] #št. osebkov
    while N>0:
        p = np.random.poisson(beta*N*dt)
        N = N - p
        t += dt
        pop.append(N)
    time = np.linspace(0,len(pop)-1,len(pop))*dt #np.arange(0.,t,dt) #časovni seznam arange nestabilen
    return time, np.array(pop)
def izumrtje(N0,beta,dt=1):
    t=0
    N=N0
    while N>0:
        p = np.random.poisson(beta*N*dt)
        N = N - p
        t += dt
    return t
def smrti_rojstva(N0,beta_s,beta_r,dt=1):
    tmax=5e3 * dt #max 10^4 iteracij
    t=0
    N=N0
    pop = [N0] #št. osebkov
    while N>0 and t<tmax:
        p = np.random.poisson(beta_r*N*dt) - np.random.poisson(beta_s*N*dt) # + rojstva - smrti
        N = N + p
        t += dt
        pop.append(N)
    time = np.linspace(0,len(pop)-1,len(pop))*dt #časovni seznam
    return time, np.array(pop)
def run_simulation(args):
    chunk_id, tasks, ratio, N0, dt = args
    results = []
    for i in tasks:
        t, N = smrti_rojstva(N0, 1, ratio, dt)
        results.append((i, t, N))
    return results

def exp_model(t, N, beta_s, beta_r=0):#za num integracijo
    return -beta_s*N + beta_r*N
def analitično(t,N0,beta_s,beta_r):
    return N0 * np.exp((beta_r-beta_s)*t)

HighPriority()

if __name__ == '__main__':
    ###########################
    # poissonska porazdelitev
    # t = np.linspace(0, 50, 1000)
    # l=10
    # d = (np.exp(-l)*np.power(l, t))/factorial(t)
    # plt.plot(t, d, '-')
    # plt.grid()
    # plt.xlabel('$x$')
    # plt.ylabel('$p(x)$')
    # # plt.savefig('./slike/poisson.pdf')
    # plt.show()

    ############################
    # potek populacije

    # N0 = 250
    # dt = 0.01
    # time_unit = 1 #enota časa - brezdim?
    # beta = 1./time_unit #podatek
    # beta_roj=0.

    # ratio = np.linspace(0,1,5) #razm. rojstva / smrti
    # beta = 1. #brezdim. smrti
    # beta_roj=ratio

    # t1, N1 = smrti_rojstva(N0,beta,beta_roj[0],dt)
    # t2, N2 = smrti_rojstva(N0,beta,beta_roj[1],dt)
    # t3, N3 = smrti_rojstva(N0,beta,beta_roj[2],dt)
    # t4, N4 = smrti_rojstva(N0,beta,beta_roj[3],dt)
    # t5, N5 = smrti_rojstva(N0,beta,beta_roj[4],dt)

    # # časovna integracija
    # # tmax=max(t1.max(),t2.max(),t3.max(),t4.max(),t5.max())
    # # t_num = np.arange(0,tmax,dt)
    # # res = solve_ivp(exp_model, y0=(N0,), t_span=(0,tmax), t_eval=t_num, args=(beta,beta_roj), method='RK45')
    # # t_num = res.t
    # # N_num = res.y[0]
    # # # N_num1 = odeint(exp_model, y0=(N0,), t=t_num, args=(beta,), tfirst=True)[:,0] #isto
    # # N_analit = analitično(t_num,N0,beta,beta_roj)

    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,4))

    # ax1.plot(t1, N1,'-',label=f'${ratio[0]}$')
    # ax1.plot(t2, N2,'-',label=f'${ratio[1]}$')
    # ax1.plot(t3, N3,'-',label=f'${ratio[2]}$')
    # ax1.plot(t4, N4,'-',label=f'${ratio[3]}$')
    # ax1.plot(t5, N5,'-',label=f'${ratio[4]}$')
    # # ax1.plot(t_num, N_num, '--', c='k', label='integracija')
    # # # ax1.plot(t_num, N_num1, '--', label='odeint')
    # # ax1.plot(t_num, N_analit, 'k-', label='analitično')
    # ax1.grid()
    # ax1.legend(title='razmerje $\\beta_r$ / $\\beta_s$')
    # ax1.set_title(f'izumiranje + rojstva $N_0={N0}$, $\Delta t={dt}$')
    # ax1.set_xlabel('$t$')
    # ax1.set_ylabel('$N$')

    # ax2.plot(t1, N1,'-')
    # ax2.plot(t2, N2,'-')
    # ax2.plot(t3, N3,'-')
    # ax2.plot(t4, N4,'-')
    # ax2.plot(t5, N5,'-')
    # # ax2.plot(t_num, N_num, '--', c='k', label='integracija')
    # # # ax2.plot(t_num, N_num1, '--', label='odeint')
    # # ax2.plot(t_num, N_analit, 'k-', label='analitično')
    # ax2.grid()
    # ax2.set_yscale('log')
    # ax2.set_title(f'izumiranje + rojstva $N_0={N0}$, $\Delta t={dt}$')
    # ax2.set_xlabel('$t$')
    # ax2.set_ylabel('$N$')

    # # ax3.plot(t1, abs(N1-analitično(t1,N0,beta,beta_roj)),'-')
    # # ax3.plot(t2, abs(N2-analitično(t2,N0,beta,beta_roj)),'-')
    # # ax3.plot(t3, abs( N3-analitično(t3,N0,beta,beta_roj)),'-')
    # # ax3.plot(t4, abs(N4-analitično(t4,N0,beta,beta_roj)),'-')
    # # ax3.plot(t5, abs(N5-analitično(t5,N0,beta,beta_roj)),'-')
    # # ax3.plot(t_num, abs( N_num-N_analit), '--', c='k', label='integracija')
    # # ax3.grid()
    # # ax3.set_yscale('log')
    # # ax3.set_title('odstopanje od analitične rešitve')
    # # ax3.set_xlabel('$t$')
    # # ax3.set_ylabel('$N$')
    # # plt.savefig(f'./slike/izumiranje_rojstva_N0{N0}.pdf')
    # plt.show()

    ###########################
    # statistika izumrtja

    # dt = 0.1
    # N0 = 250
    # n_max = 500000 #št izračunov OK
    # n_max = 10000 #ok povprecje, a grav ni lep

    # čas = timer()
    # beta_1 = 1.
    # time_1 = [] #časi izumrtja
    # for i in range(n_max):
    #     t= izumrtje(N0,beta_1,dt)
    #     time_1.append(t)
    # čas_1=timer()
    # print('t_sim = ',čas_1-čas)
    # beta_5 = 0.5
    # time_5 = [] #časi izumrtja
    # for i in range(n_max):
    #     t= izumrtje(N0,beta_5,dt)
    #     time_5.append(t)
    # čas_2=timer()
    # print('t_sim = ',čas_2-čas_1)
    # beta_25 = 0.25
    # time_25 = [] #časi izumrtja
    # for i in range(n_max):
    #     t= izumrtje(N0,beta_25,dt)
    #     time_25.append(t)
    # čas_3=timer()
    # print('t_sim = ',čas_3-čas_2)

    # fig,ax=plt.subplots()
    # plt.hist(time_1, bins=int((max(time_1)-min(time_1))/dt), edgecolor='k', fill=False, histtype='step', lw=2, label=beta_1)
    # plt.hist(time_5, bins=int((max(time_5)-min(time_5))/dt), edgecolor='g', fill=False, histtype='step', lw=2, label=beta_5)
    # plt.hist(time_25, bins=int((max(time_25)-min(time_25))/dt), edgecolor='b', fill=False, histtype='step', lw=2, label=beta_25)

    # avg_1 = np.average(time_1)
    # avg_5 = np.average(time_5)
    # avg_25 = np.average(time_25)
    # plt.axvline(avg_1, c='k', ls='--')
    # plt.axvline(avg_5, c='g', ls='--')
    # plt.axvline(avg_25, c='b', ls='--')
    # tmax=max(max(time_1), max(time_5), max(time_25))
    # plt.text(avg_1/tmax+0.03, 1, f'$t_{{avg}}={avg_1:.2f}$', c='k', horizontalalignment='left', verticalalignment='top', rotation='vertical',transform = ax.transAxes)
    # plt.text(avg_5/tmax+0.03, 1, f'$t_{{avg}}={avg_5:.2f}$', c='g', horizontalalignment='left', verticalalignment='top', rotation='vertical',transform = ax.transAxes)
    # plt.text(avg_25/tmax+0.02, 1, f'$t_{{avg}}={avg_25:.2f}$', c='b', horizontalalignment='left', verticalalignment='top', rotation='vertical',transform = ax.transAxes)
    # #mean=average za Poissona?

    # plt.grid()
    # plt.legend(title='$\\beta$')
    # plt.xlabel('$t_{izumrtje}$')
    # plt.ylabel('$N_{ponovitev}$')
    # plt.title(f'$N_0={N0}$, $\Delta t$ = {dt}')
    # # plt.savefig(f'./slike/t_izumrtja_N0{N0}.pdf')
    # plt.show()

    #########################
    # verjetnost poteka populacije

    N0 = 250
    ratio = 0. #razmerje rojstva / smrti
    beta = 1 #smrti
    beta_roj = ratio #rojstva
    dt = 0.01
    n_max = 10000
    
    # ------------- 1. zaporedno
    # t1=timer()
    # time=[]
    # pop=[]
    # err=0
    # for i in tqdm(range(n_max)):
    #     # t,N = smrti(N0,beta,dt)
    #     t,N = smrti_rojstva(N0,beta,beta_roj,dt)
    #     time.append(t)
    #     pop.append(N)
    #     if len(t) != len(N):
    #         err+=1
    #         # print(f'Časovni in populacijski senam nista enako dolga! Iteracija: {i}, {err} napak.',end='\r')
    # print(f'Časovni in populacijski senam nista enako dolga {100*err/n_max} % pogosto!')
    # t2=timer()
    # print(f'Vzporedno: {t2-t1} s')

    # ------------ 2. vzporedno -> POČASNEJE za veliko majhnih procesov!
    max_workers = 18
    chunk_size = n_max//max_workers # max iteracij na vsak proces da se ne zaganja novih!

    t1=timer()
    time=[]
    pop=[]
    tasks = list(range(n_max))
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
    results = []
    t11=timer()
    print(f'Vzporedno priprava: {t11-t1} s')

    with ProcessPoolExecutor(max_workers) as executor:
        # Initialize the progress bar
        progress_bar = tqdm(total=len(chunks), desc='Processing')

        # Function to update the progress bar
        def update_progress(*args):
            progress_bar.update()

        # Create a list of futures
        futures = [executor.submit(run_simulation, (chunk_id, chunk, ratio,N0,dt)) for chunk_id, chunk in enumerate(chunks)]
        for future in futures:
            future.add_done_callback(update_progress)

        # Wait for all futures to complete and collect results
        for future in futures:
            results.extend(future.result())

        # Close the progress bar
        progress_bar.close()
    t2=timer()
    print(f'Vzporedno sim: {t2-t1} s')

    # Store results in t_list
    for i, t, N in results:
        time.append(t)
        pop.append(N)
    t3=timer()
    print(f'Vzporedno post: {t3-t2} s')

    # -------------------------------- obdelava rezultatov
    tmax = max([len(i) for i in time])
    Nmax = max([max(i) for i in pop])
    data = np.zeros((tmax, Nmax+1)) #max(N)=N0 za smrti
    data[::] = 1e-12 #numerika

    # for i in range(len(t)):
    #     for j in range(max(N)+1):
    #         if j == N[i]:
    #             data[i,j] += 1
    for i in range(n_max): #hitreje!
        t_pos=np.arange(0,len(time[i]),1,dtype=int) #pozicije časov v matriki
        p_pos=np.array(pop[i],dtype=int)
        data[t_pos[:len(p_pos)],p_pos] += 1
    data = data/np.max(data) #scale

    # časovna integracija
    t_num = np.arange(0,tmax*dt,dt)
    res = solve_ivp(exp_model, y0=(N0,), t_span=(0,t_num[-1]), t_eval=t_num, args=(beta, beta_roj), method='RK45')
    t_num = res.t
    N_num = res.y[0]

    fig, ax = plt.subplots()
    im=ax.imshow(data.T,origin='lower', cmap='Paired', aspect='auto', norm=LogNorm(vmin=1e-12,vmax=1.))
    cbar=fig.colorbar(im,ax=ax,label='pogostost')
    
    num_xticks = 6
    xtick_positions = np.linspace(0, tmax - 1, num_xticks, dtype=int)
    xticks = [f'{np.arange(0,tmax,dt)[i]:.2f}' for i in xtick_positions]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticks)
    # ytick_positions = np.linspace(0, Nmax - 1, num_xticks, dtype=int)
    # yticks = [f'{np.arange(0,Nmax)[i]:.2f}' for i in ytick_positions]
    # ax.set_yticks(ytick_positions)
    # ax.set_yticklabels(yticks)

    # ax.plot(t_num, N_num, 'w:', label='IVP') #povzroča težave
    # ax.legend()

    plt.xlabel('$t$')
    plt.ylabel('$N$')
    plt.title(f'brez rojstev $\\beta={beta}$, $\Delta t={dt}$, $N_0={N0}$, {n_max} ponovitev') #le smrti
    plt.title(f'$ratio={ratio}$ $\Delta t={dt}$, $N_0={N0}$, {n_max} ponovitev')
    # plt.savefig(f'./slike/verjetnost_poteka_ratio{ratio}.pdf')
    plt.show()

    #########################
    # odvisnost dt, beta - čas izumrtja -> nov file!