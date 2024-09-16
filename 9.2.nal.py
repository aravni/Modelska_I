# -*- coding: utf-8 -*-
# stohastični populacijski modeli: formulacija z matriko prehodov

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from ProcessPriority import *
from matplotlib.animation import FuncAnimation, PillowWriter, MovieWriter

HighPriority()

N = 100 #št. nivojev
ratio = 0. #razmerje rojstva / smrti
n_max_iters = 10000 #max št. časovnih korakov
dt = 0.005 #čas korak

beta_s=1. # brezdim: smrti = 1
beta_r=ratio*beta_s
levels=np.array(range(N)) #nivoji
time = np.linspace(0,n_max_iters,n_max_iters+1) *dt #np.arange(0,(n_max_iters+2)*dt,dt) #num nestabilna dolžina!
M = np.zeros((N,N)) #matrika prehodov
X = np.zeros((n_max_iters+1,N)) #vektor vektorjev stanj
var = np.zeros((n_max_iters+1))
skew = np.zeros((n_max_iters+1))
kurtosis = np.zeros((n_max_iters+1))
avg = np.zeros((n_max_iters+1))

# prvi korak: S_0 = R_0 = 0, zač. hitrost rojstev/smrti je nič!
# R_1 = 1*beta_r, S_1 = 1*beta_s
M[0,0] += 1 #- dt*beta_r*0 - beta_s*0*dt
M[0,1] += beta_s*dt

for n in range(1,N-1):
    R_n = beta_r*n #definicija
    S_n = beta_s*n
    R_n_1 = beta_r*(n-1)
    S_n1 = beta_s*(n+1)
    M[n,n-1] += R_n_1*dt #R_n-1
    M[n,n+1] += S_n1*dt #S_n+1
    M[n,n] += 1 - R_n*dt - S_n*dt

#zadnja vrstica -> čudno šteteje Python?
R_N_1 = beta_r*(N-2) #R_{N-1}
R_N = beta_r*(N-1)
S_N = beta_s*(N-1)
M[-1,-2] += R_N_1*dt
M[-1,-1] += 1 - S_N*dt #- R_N*dt na koncu ni rojstev?!

for i in range(N):#check
    stolpec = np.sum(M[:,i])
    if np.isclose(stolpec, 1) == False: print(f'Napaka: vsota stolpca {i} je {stolpec} ni enako 1 ! Napaka {stolpec-1.}')
plt.imshow(M,origin='upper',cmap='binary')
plt.colorbar()
plt.title('matrika prehodov') #sistem je stabilen, če je matrika diagonalno dominatna !!!
plt.show()

labels=np.linspace(0,100,11, dtype=int)
ticks=np.linspace(0,1,11)

#ZP
x0 = np.zeros(N) # def: sum(x) = 1
x0[-1] = 1 #ZP, zaseden le najvišji nivo
X[0,:] = x0
var[0] = np.sum(levels**2*x0)
skew[0] = np.sum(levels**3*x0)
kurtosis[0] = np.sum(levels**4*x0)
avg[0] = x0[-1]*np.exp(-(beta_r-beta_s)*0) #t_0=0

#######################
# 'ročna' animacija
t1=timer()

# plt.ion()
# fig, ax = plt.subplots()
# ax.bar(levels,x0)
# ax.set_xlim(0,N)
# ax.set_ylim(0,1)
# ax.set_yticks(ticks=ticks, labels=labels)
# ax.set_xlabel('nivoji $x_n$')
# ax.set_ylabel('delež populacije [%]')
# ax.set_title(f'$t=0$')
# ax.grid()
# plt.show()
# plt.pause(0.001)

# prvi korak iz ZP
x = M@x0
X[1,:] = x
var[1] = np.sum(levels**2*x)
skew[1] = np.sum(levels**3*x)
kurtosis[1] = np.sum(levels**4*x)
avg[1] = x0[-1]*np.exp(-(beta_r-beta_s)*dt)

# ax.cla()
# ax.bar(levels,x)
# ax.set_xlim(0,N)
# ax.set_ylim(0,1)
# ax.set_yticks(ticks=ticks, labels=labels)
# ax.set_xlabel('nivoji $x_n$')
# ax.set_ylabel('delež populacije [%]')
# ax.set_title(f'$t={dt:.3f}$')
# ax.grid()
# plt.show()
# plt.pause(.001)

n_iter = 1
while x[0] < 0.99 and n_iter < n_max_iters:
    n_iter+=1
    x = M@x
    X[n_iter,:] = x
    var[n_iter] = np.sum(levels**2*x) #ročna 0. in 1. korak
    skew[n_iter] = np.sum(levels**3*x)
    kurtosis[n_iter] = np.sum(levels**4*x)
    avg[n_iter] = x0[-1]*np.exp(-(beta_r-beta_s)*(n_iter)*dt)

#     ax.cla()
#     ax.bar(levels,x)
#     ax.set_xlim(0,N)
#     ax.set_ylim(0,1)
#     ax.set_yticks(ticks=ticks, labels=labels)
#     ax.set_xlabel('nivoji $x_n$')
#     ax.set_ylabel('delež populacije [%]')
#     ax.set_title(f'$t={dt*(n_iter):.3f}$')
#     ax.grid()
#     plt.show()
#     plt.pause(.001)

# plt.ioff()
# plt.show()
t2=timer()
print(f't_sim = {t2-t1}')

#########################
# shraniti animacijo

fig, ax = plt.subplots()

def update(frame):
    ax.cla()
    ax.bar(levels, X[frame,:])
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)
    ax.set_yticks(ticks=ticks, labels=labels)
    ax.set_xlabel('nivoji $x_n$')
    ax.set_ylabel('delež populacije [%]')
    ax.set_title(f'$t={dt * frame:.3f}$')
    ax.grid()

step = n_iter//60
print(n_iter, step, n_iter/step)
# ani = FuncAnimation(fig, update, frames=range(0,n_iter,step), repeat=False, interval=1e-9) #vemo pribl. koliko iteracij je potrebnih! - NAKLJUČNOST
# ani.save(f"./slike/nivoji{N}_ratio{ratio}.gif", writer=PillowWriter(fps=60), dpi=150)
# t3=timer()
# print(f'čas za prikaz = {t3-t2}')
# plt.show()

###################
# statistika

std = np.sqrt(var[:n_iter+1]) #stand odklon
skew = skew[:n_iter+1]/std**3
kurtosis = kurtosis[:n_iter+1]/std**4
avg = avg[:n_iter+1]
time = time[:n_iter+1]

fig,ax=plt.subplots()
# ax.plot(time, avg, c='b',label='srednja vrednost')
# plt.plot(time, var, label='varianca')
ax.plot(time, std, c='g', label='standardni odklon')
ax.plot(time, skew, c='orange', label='skewness/asimetričnost')

ax1=plt.twinx()
ax1.set_ylabel('kurtosis/sploščenost', c='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.plot(time, kurtosis, c='r',label='kurtosis/sploščenost')

ax.legend(loc='upper center')
ax.grid()
plt.title(f'statistične veličine, $ratio={ratio}$, $\Delta t ={dt}$')
plt.tight_layout()
# plt.savefig(f'./slike/stat_vel_ratio{ratio}.pdf')
plt.show()

# primerjava s poissonom
def exp_analit(t,N0,beta_s,beta_r):
    return N0 * np.exp((beta_r-beta_s)*t)
def smrti_rojstva(N0,beta_s,beta_r,dt=1,n_max_iters=1e4):
    tmax=n_max_iters * dt #max 10^4 iteracij
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
def model(N, ratio, n_max_iters, dt):
    beta_s=1. # brezdim: smrti = 1
    beta_r=ratio*beta_s
    M = np.zeros((N,N)) #matrika prehodov
    # X = np.zeros((n_max_iters+1, N))
    avg = np.zeros((n_max_iters+1))

    # prvi korak: S_0 = R_0 = 0, zač. hitrost rojstev/smrti je nič!
    # R_1 = 1*beta_r, S_1 = 1*beta_s
    M[0,0] += 1 #- dt*beta_r*0 - beta_s*0*dt
    M[0,1] += beta_s*dt

    for n in range(1,N-1):
        R_n = beta_r*n #definicija
        S_n = beta_s*n
        R_n_1 = beta_r*(n-1)
        S_n1 = beta_s*(n+1)
        M[n,n-1] += R_n_1*dt #R_n-1
        M[n,n+1] += S_n1*dt #S_n+1
        M[n,n] += 1 - R_n*dt - S_n*dt

    #zadnja vrstica -> čudno šteteje Python?
    R_N_1 = beta_r*(N-2) #R_{N-1}
    S_N = beta_s*(N-1)
    M[-1,-2] += R_N_1*dt
    M[-1,-1] += 1 - S_N*dt #- R_N*dt na koncu ni rojstev?!

    for i in range(N):#check
        stolpec = np.sum(M[:,i])
        if np.isclose(stolpec, 1) == False: print(f'Napaka: vsota stolpca {i} je {stolpec} ni enako 1 ! Napaka {stolpec-1.}')

    #ZP
    x0 = np.zeros(N) # def: sum(x) = 1
    x0[-1] = 1 #ZP, zaseden le najvišji nivo
    x = x0
    avg[0] = x0[-1] # np.exp((-beta_r - beta_s) * 0 * dt) = 1

    n_iter = 1
    while x[0] < 0.99 and n_iter < n_max_iters:
        x = M@x
        avg[n_iter] = x0[-1] * np.exp((-beta_r - beta_s) * (n_iter) * dt)
        n_iter+=1

    if n_iter == n_max_iters: print('max iteracije!')
    # X = X[:n_iter+1,:]
    avg = np.array(avg[:n_iter+1])
    time = np.linspace(0,n_iter,n_iter+1)*dt
    return avg, time

N = 100 #št. nivojev
ratio = 0. #razmerje rojstva / smrti
n_max_iters = 10000 #max št. časovnih korakov
dt = 0.005 #čas korak

N_avg, t = model(N, ratio, n_max_iters, dt) #enako kot prej!

analiticno = exp_analit(t,x0[-1], beta_s, beta_r)

N0=100
n_ponovitev=100
N_p=np.zeros((n_ponovitev, n_max_iters))
for i in range(n_ponovitev):
    t_p, N_pi = smrti_rojstva(N0,beta_s,beta_r,dt,n_max_iters)
    N_p[i,:len(N_pi)] = N_pi
N_p_avg = np.array([np.average(N_p[:,i]) for i in range(n_max_iters)])
N_p_avg = N_p_avg[N_p_avg != 0]
N_p_avg = np.append(N_p_avg, 0)
t_p = np.linspace(0,len(N_p_avg), len(N_p_avg))*dt

# plt.plot(time, N0*avg, 'C0', label='matrično')
plt.plot(t, N0*N_avg,'C0', label='matrično')
plt.axvline(t[-1], c='C0',ls='--')
plt.text(t[-1], 1, f'$t=${t[-1]:.3f}', c='C0', horizontalalignment='left', verticalalignment='bottom', rotation='vertical')
# plt.fill_between(time, avg+std, avg-std, color='C0', alpha=0.7)
plt.plot(t, N0*analiticno, 'C1--', label='analitično')
plt.plot(t_p, N_p_avg, 'C2', label='povprečen Poisson')
plt.axvline(t_p[-1], c='C2',ls='--')
plt.text(t_p[-1], 1, f'$t=${t_p[-1]:.3f}', c='C2', horizontalalignment='left', verticalalignment='bottom', rotation='vertical')
plt.xlabel('$t$')
plt.ylabel('delež populacije [%]')
plt.title(f'$ratio={ratio}$, $\Delta t ={dt}$')
plt.grid()
plt.legend()
plt.yscale('log')
plt.tight_layout()
# plt.savefig(f'./slike/primerjava_ratio{ratio}.pdf')
plt.show()