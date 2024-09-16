# -*- coding: utf-8 -*-
# stohastični populacijski modeli: model epidemije

import numpy as np
from numpy.random import poisson
import matplotlib.pyplot as plt 
from timeit import default_timer as timer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ProcessPriority import *
from scipy.integrate import solve_ivp
import pandas as pd

def epidemija(N0,B0,R,dt,n_max_iters,I0=0):
    B=B0 #bolni
    I=I0 #precepljenost
    D=int(N0-B0-I0) #dovzetni
    n_iter=0
    bolni = [B0]
    imuni = [I0]
    dovzetni = [D]
    while n_iter<n_max_iters and B>0:
        oboleli = poisson(R*D*B/N0*dt)
        preboleli = poisson(B*dt)
        B = B + oboleli - preboleli
        I = I + preboleli
        D = D - oboleli
        n_iter += 1
        bolni.append(B)
        imuni.append(I)
        dovzetni.append(D)
    t = np.linspace(0,len(bolni)-1, len(bolni))*dt
    return np.array(bolni), np.array(imuni), np.array(dovzetni), t
def epidemija_num(t, y, R):
    d, b, i = y
    return [-R*d*b, R*d*b - b, b]

def epidemija_ozdravitev(N0,B0,R,Ir,dt,n_max_iters,I0=0):
    '''imuni prehajajo s konstatno Ir nazaj v populacijo dovzetnih'''
    B=B0 #bolni
    I=I0 #precepljenost
    D=int(N0-B0-I0) #dovzetni
    n_iter=0
    bolni = [B0]
    imuni = [I0]
    dovzetni = [D]
    while n_iter<n_max_iters and B>0:
        oboleli = poisson(R*D*B/N0*dt)
        preboleli = poisson(B*dt)
        ozdraveli = poisson(Ir*I*dt)
        B = B + oboleli - preboleli
        I = I + preboleli - ozdraveli
        D = D - oboleli + ozdraveli
        n_iter += 1
        bolni.append(B)
        imuni.append(I)
        dovzetni.append(D)
    t = np.linspace(0,len(bolni)-1, len(bolni))*dt
    return np.array(bolni), np.array(imuni), np.array(dovzetni), t
def ozdravitev_num(t, y, R, I):
    d,b,i = y #stanje
    return [-R*d*b + I*i, R*d*b - b, b - I*i]

def epidemija_r_s(N0,B0,R,Ir,Mr,Nr,dt,n_max_iters,I0=0):
    '''rojstva+smrti'''
    B=B0 #bolni
    I=I0 #precepljenost-imuni
    D=int(N0-B0-I0) #dovzetni
    M=0 #umrli
    n_iter=0
    bolni = [B0]
    imuni = [I0]
    dovzetni = [D]
    mrtvi=[M]
    while n_iter<n_max_iters and B>0:
        oboleli = poisson(R*D*B/N0*dt)
        preboleli = poisson(B*dt)
        ozdraveli = poisson(Ir*I*dt)
        rojeni = poisson(Nr*(I+D)*dt)
        umrli = poisson(Mr*B*dt)
        B = B + oboleli - preboleli - umrli
        I = I + preboleli - ozdraveli
        D = D - oboleli + ozdraveli + rojeni
        M = M + umrli
        n_iter += 1
        bolni.append(B)
        imuni.append(I)
        dovzetni.append(D)
        mrtvi.append(M)
    t = np.linspace(0,len(bolni)-1, len(bolni))*dt
    return np.array(bolni), np.array(imuni), np.array(dovzetni), np.array(mrtvi), t
def r_s_num(t,y,R,I,M=0, N=0):#rojstva+smrti
    d,b,i,m = y
    return [-R*d*b + I*i + N*(d+i), R*d*b - b - M*b, b - I*i, M*b]

N0=1000 #populacija
B0=N0//100 #okuženi
I0=0 #precepljeni
Rr = 10 #reprodukcijsko število (razmerje) = alpha/beta
Ir = 0.01 #razmerje imunosti = gamma/beta
dt=0.01
n_max_iters = 100000

# B,I,D, t = epidemija(N0,B0,Rr,dt,n_max_iters,I0=I0)
B,I,D, t = epidemija_ozdravitev(N0,B0,Rr,Ir,dt,n_max_iters,I0=I0)
res = solve_ivp(ozdravitev_num, y0=((N0-B0-I0)/N0,B0/N0,I0/N0), t_span=(t[0],t[-1]), args=(Rr,Ir), t_eval=t, method='RK45')
D_n, B_n, I_n = res.y

# plt.plot(t,B,label='bolni')
# plt.plot(t,D,label='dovzetni')
# plt.plot(t,I,label='imuni')
# plt.legend()
# plt.grid()
# plt.show()

lega_max = np.where(B == max(B))[0][0]
print(f'vrh epidemije {100*B[lega_max]/N0} %, pri  t = {round(t[lega_max],3)}, {lega_max} iteracij')
print(f'skupno število obolelih: {int(sum(B))}, skupaj {len(t)-1} iteracij')

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
df = pd.DataFrame({
    'bolni': 100*B/N0,
    'dovzetni' : 100*D/N0,
    'imuni': 100*I/N0
}, index=t)
df.plot.area(ax=ax1,)
ax1.set_xlabel('$t$')
ax1.set_ylabel('$\%$ populacije')
ax1.plot(t, 100*D*B/N0**2, 'k--', label='izpostavljeni')
ax1.get_legend().remove()
ax1.set_title('stohastično')
ax1.text(t[lega_max], 5+B[lega_max]/N0*100, f'vrh: $t = {t[lega_max]:.2f}$, {100*B[lega_max]/N0:.0f}% obolelih', bbox=dict(facecolor='white', alpha=0.5))

lega_max = np.where(B_n == max(B_n))[0][0]
df = pd.DataFrame({
    'bolni': 100*B_n,
    'dovzetni' : 100*D_n,
    'imuni': 100*I_n
}, index=t)
df.plot.area(ax=ax2)
ax2.set_xlabel('$t$')
ax2.set_ylabel('$\%$ populacije')
ax2.plot(t, 100*D_n*B_n, 'k--', label='izpostavljeni')
ax2.legend(bbox_to_anchor=(1, 1), ncol=1)
ax2.set_title('numerična integracija')
ax2.text(t[lega_max], 5+B_n[lega_max]*100, f'vrh: $t = {t[lega_max]:.2f}$, {100*B_n[lega_max]:.0f}% obolelih', bbox=dict(facecolor='white', alpha=0.5))
plt.suptitle(f'$\Delta t ={dt}$, $R_r={Rr}$, $I_r={Ir}$, $B_0={N0//B0//100}$%')
plt.tight_layout()
# plt.savefig('./slike/potek_ozdravitev.pdf')
plt.show()

# plt.plot(t,D,'C0',label='dovzetni')
# plt.plot(t,B,'C1',label='bolni')
# plt.plot(t,I,'C2',label='imuni')
# plt.plot(t,D+B+I,'C3')

# plt.plot(t,N0*D_n,'C0--')
# plt.plot(t,N0*B_n,'C1--')
# plt.plot(t,N0*I_n,'C2--')
# plt.plot(t,N0*(D_n+B_n+I_n),'C3--')

# plt.legend()
# plt.show()

###### rojstva + smrti
# Rr = 1
# Ir = 1
Mr = 0.1 # razm. smrti
Nr = 0.01 # razm. rojstev

B,I,D,M, t = epidemija_r_s(N0,B0,Rr,Ir,Mr,Nr,dt,n_max_iters,I0=I0)
res = solve_ivp(r_s_num, y0=((N0-B0-I0)/N0,B0/N0,I0/N0,0.), t_span=(t[0],t[-1]), args=(Rr,Ir,Mr,Nr), t_eval=t, method='RK45')
D_n, B_n, I_n, M_n = res.y

lega_max = np.where(B == max(B))[0][0]
print(f'vrh epidemije {100*B[lega_max]/N0} %, pri  t = {round(t[lega_max],3)}, {lega_max} iteracij')
print(f'skupno število obolelih: {int(sum(B))}, skupaj {len(t)-1} iteracij')

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
df = pd.DataFrame({
    'bolni': B/N0*100,
    'dovzetni' : D/N0*100,
    'imuni': I/N0*100,
    'umrli': M/N0*100,
}, index=t)
df.plot.area(ax=ax1)
ax1.set_xlabel('čas $t$')
ax1.set_ylabel('$\%$ populacije')
ax1.plot(t, 100*D*B/N0**2, 'k--', label='izpostavljeni')
ax1.get_legend().remove()
ax1.set_title('stohastično')
ax1.text(t[lega_max], 5+B[lega_max]/N0*100, f'vrh: $t = {t[lega_max]:.2f}$, {100*B[lega_max]/N0:.0f}% obolelih', bbox=dict(facecolor='white', alpha=0.5))

df = pd.DataFrame({
    'bolni': 100*B_n,
    'dovzetni' : 100*D_n,
    'imuni': 100*I_n,
    'umrli': 100*M_n,
}, index=t)
df.plot.area(ax=ax2)
ax2.set_xlabel('čas $t$')
ax2.set_ylabel('$\%$ populacije')
ax2.plot(t, 100*D_n*B_n, 'k--', label='izpostavljeni')
ax2.legend(bbox_to_anchor=(1, 1), ncol=1)
ax2.set_title('numerična integracija')
ax2.text(t[lega_max], 5+B_n[lega_max]*100, f'vrh: $t = {t[lega_max]:.2f}$, {100*B_n[lega_max]:.0f}% obolelih', bbox=dict(facecolor='white', alpha=0.5))
plt.suptitle(f'$\Delta t ={dt}$, $R_r={Rr}$, $I_r={Ir}$, $M_r={Mr}$, $N_r={Nr}$,$N_0={N0}$, $B_0={N0//B0//100}$%')
plt.tight_layout()
# plt.savefig('./slike/potek_smrti_rojstva.pdf')
plt.show()

######################
# statistika
dt=0.01
n_max_iters = 1e5
N0=1000
B0=N0/100
n_ponovitev = 20
print('t_max', dt*n_max_iters)
num_xticks = num_yticks = 5

R_list = np.linspace(0.1,10,10)
I_list = np.linspace(0.001,0.01,10)

############## 1. določanje vrhov epidemije
# vrh = np.zeros((len(R_list),len(I_list)))
# for i in tqdm(range(len(R_list))):
#     Rr=R_list[i]
#     for j in range(len(I_list)):
#         Ir=I_list[j]
#         v=np.zeros((n_ponovitev))
#         for k in range(n_ponovitev):
#             try:
#                 B,I,D,t = epidemija_ozdravitev(N0,B0,Rr,Ir,dt,n_max_iters)
#                 v[k] = max(B)
#             except:
#                 v[k] = 0
#         vrh[i,j] = np.average(v)

# # vrh epidemija
# plt.imshow(vrh/N0*100)
# plt.colorbar(label='vrh epidemije [% populacije]')

# # Determine the tick positions and labels
# xtick_positions = np.linspace(0, len(R_list) - 1, num_xticks, dtype=int)
# xticks = [f'{R_list[i]:.2f}' for i in xtick_positions]

# ytick_positions = np.linspace(0, len(I_list) - 1, num_yticks, dtype=int)
# yticks = [f'{I_list[i]:.3f}' for i in ytick_positions]

# # Set the tick positions and labels for the x-axis
# plt.gca().set_xticks(xtick_positions)
# plt.gca().set_xticklabels(xticks)

# # Set the tick positions and labels for the y-axis
# plt.gca().set_yticks(ytick_positions)
# plt.gca().set_yticklabels(yticks)

# plt.xlabel('parameter $R$')
# plt.ylabel('parameter $I$')
# # plt.savefig('./slike/vrh_epidemija_cmap.pdf')
# plt.show()

############ 2. čas izumrtja

# parameter R
# n_ponovitev = 1000
# Ir = 0.001
# R_list = np.linspace(0.1,10,20)
# t_izumrt_R = np.zeros((len(R_list),n_ponovitev))
# colors = plt.cm.viridis(np.linspace(0, 1, len(R_list)))
# fig, ax = plt.subplots()
# for i in tqdm(range(len(R_list))):
#     Rr=R_list[i]
#     for j in range(n_ponovitev):
#         try:
#             B,I,D,t = epidemija_ozdravitev(N0,B0,Rr,Ir,dt,n_max_iters)
#             if len(t)==n_max_iters+1:
#                 print(' max iteracije')
#                 t_izumrt_R[i,j] = 0
#             else:
#                 t_izumrt_R[i,j] = t[-1]
#         except:
#             t_izumrt_R[i,j] = 0
#     ax.hist(t_izumrt_R[i, :], bins=100, alpha=0.6, color=colors[i], label=f'{Rr:.1f}')

# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(R_list), vmax=max(R_list)))
# sm.set_array([])  # You can set the array to None as it's not used directly
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('R parameter')
# plt.xlabel('$t_{izumrtja}$')
# plt.ylabel('$N$')
# plt.grid()
# # plt.savefig(f'./slike/hist_R.pdf')
# plt.show()

# parameter I
# n_ponovitev = 1000
# R_r=5
# I_list = np.linspace(0.001,0.01,20)
# t_izumrt_I = np.zeros((len(I_list),n_ponovitev))
# colors = plt.cm.viridis(np.linspace(0, 1, len(I_list)))
# fig, ax = plt.subplots()
# for i in tqdm(range(len(I_list))):
#     Ir=I_list[i]
#     for j in range(n_ponovitev):
#         try:
#             B,I,D,t = epidemija_ozdravitev(N0,B0,Rr,Ir,dt,n_max_iters)
#             if len(t)==n_max_iters+1:
#                 print(' max iteracije')
#                 t_izumrt_I[i,j] = 0
#             else:
#                 t_izumrt_I[i,j] = t[-1]
#         except:
#             t_izumrt_I[i,j] = 0
#     ax.hist(t_izumrt_I[i, :], bins=100, alpha=0.6, color=colors[i])

# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1e3*min(I_list), vmax=1e3*max(I_list)))
# sm.set_array([])  # You can set the array to None as it's not used directly
# cbar = plt.colorbar(sm, ax=ax)
# cbar.set_label('I parameter $[10^{-3}]$')
# plt.xlabel('$t_{izumrtja}$')
# plt.ylabel('$N$')
# plt.grid()
# # plt.savefig(f'./slike/hist_I.pdf')
# plt.show()

############ 3. velikost populacije
n_ponovitev = 1000
R_r=5
Ir = 0.001
N0_list = np.linspace(10, 10000, 20)
t_izumrt_N0 = np.zeros((len(N0_list),n_ponovitev))
colors = plt.cm.viridis(np.linspace(0, 1, len(N0_list)))
fig, ax = plt.subplots()
for i in tqdm(range(len(N0_list))):
    N0=N0_list[i]
    for j in range(n_ponovitev):
        try:
            B,I,D,t = epidemija_ozdravitev(N0,B0,Rr,Ir,dt,n_max_iters)
            if len(t)==n_max_iters+1:
                print(' max iteracije')
                t_izumrt_N0[i,j] = 0
            else:
                t_izumrt_N0[i,j] = t[-1]
        except:
            t_izumrt_N0[i,j] = 0
    ax.hist(t_izumrt_N0[i, :], bins=100, alpha=0.6, color=colors[i])

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(N0_list), vmax=max(N0_list)))
sm.set_array([])  # You can set the array to None as it's not used directly
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('$N_0$')
plt.xlabel('$t_{izumrtja}$')
plt.ylabel('$N$')
plt.grid()
plt.savefig(f'./slike/hist_N0.pdf')
plt.show()