import numpy as np
import matplotlib.pyplot as plt

def N1_M1(x_fp, theta_fp, a):
    a0,a1,a3,a4 = a
    return (a0+a1*x_fp) + (a3+a4*x_fp)*theta_fp
def N2_M1(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5 = a
    return (a0+a1*x_fp) + (a3+a4*x_fp)*theta_fp + (a2+a5*x_fp)*theta_fp**2
def N3_M1(x_fp, theta_fp, a):
    a0,a1,a3,a4,a6,a7,a9,a10 = a
    return (a0+a1*x_fp) + (a3+a4*x_fp)*theta_fp + (a6+a7*x_fp)*theta_fp**2 + (a9+a10*x_fp)*theta_fp**3
def N4_M1(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a9,a10 = a
    return (a0+a1*x_fp) + (a3+a4*x_fp)*theta_fp + (a6+a7*x_fp)*theta_fp**2 + (a9+a10*x_fp)*theta_fp**3 + (a2+a5*x_fp)*theta_fp**4

def N1_M2(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5 = a
    return (a0+a1*x_fp+a2*x_fp**2) + (a3+a4*x_fp+a5*x_fp**2)*theta_fp
def N1_M3(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7 = a
    return (a0+a1*x_fp+a2*x_fp**2+a6*x_fp**3) + (a3+a4*x_fp+a5*x_fp+a7*x_fp**3)*theta_fp
def N1_M4(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a8,a9 = a
    return (a0+a1*x_fp+a2*x_fp**2+a6*x_fp**3+a8*x_fp**4) + (a3+a4*x_fp+a5*x_fp+a7*x_fp**3+a9*x_fp**4)*theta_fp

def N2_M2(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a8 = a
    return (a0+a1*x_fp+a2*x_fp**2) + (a3+a4*x_fp+a5*x_fp**2)*theta_fp + (a6+a7*x_fp+a8*x_fp**2)*theta_fp**2
def N3_M3(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 = a
    return (a0+a1*x_fp+a2*x_fp**2+a6*x_fp**3) + (a3+a4*x_fp+a5*x_fp+a7*x_fp**3)*theta_fp + (a8+a9*x_fp+a10*x_fp+a11*x_fp**3)*theta_fp**2 + (a12+a13*x_fp+a14*x_fp+a15*x_fp**3)*theta_fp**3
def N4_M4(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24 = a
    return (a0+a1*x_fp+a2*x_fp**2+a6*x_fp**3+a8*x_fp**4) + (a3+a4*x_fp+a5*x_fp+a7*x_fp**3+a9*x_fp**4)*theta_fp + (a10+a11*x_fp+a12*x_fp+a13*x_fp**3+a14*x_fp**4)*theta_fp**2 + (a15+a16*x_fp+a17*x_fp+a18*x_fp**3+a19*x_fp**4)*theta_fp**3 + (a20+a21*x_fp+a22*x_fp+a23*x_fp**3+a24*x_fp**4)*theta_fp**4

data = np.loadtxt("C:/Users/ravni/Documents/FS/Modelska analiza I/thtg-xfp-thfp.dat")
thtg = []
xfp = []
thfp = []
for vrstica in data:
    thtg.append(vrstica[0])
    xfp.append(vrstica[1])
    thfp.append(vrstica[2])
thtg = np.array(np.deg2rad(thtg)) #deg -> rad
xfp = np.array(xfp) #mm
thfp = np.array(np.deg2rad(thfp)) #deg -> rad
#napaka meritve
d_xfp = np.array([0.001 for i in xfp]) #mili rad
d_thfp = np.array([1 for i in thfp]) #mm

# spreminjamo red theta - N, M=1
X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, thfp, xfp * thfp))
a11, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, thfp, xfp * thfp, thfp**2, xfp * thfp**2))
a21, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, thfp, xfp * thfp, thfp**2, xfp * thfp**2, thfp**3, xfp * thfp**3))
a31, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, thfp, xfp * thfp, thfp**2, xfp * thfp**2, thfp**3, xfp * thfp**3, thfp**4, xfp * thfp**4))
a41, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

# spreminjamo red x - M, N=1
X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, thfp, xfp * thfp, xfp ** 2 * thfp))
a12, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, xfp ** 3, thfp, xfp * thfp, xfp ** 2 * thfp, xfp ** 3 *thfp))
a13, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, xfp ** 3, xfp ** 4, thfp, xfp * thfp, xfp ** 2 * thfp, xfp ** 3 * thfp, xfp ** 4 *thfp))
a14, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

# M = N
X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, thfp, xfp * thfp, xfp ** 2 * thfp, thfp, xfp * thfp**2, xfp ** 2 * thfp**2))
a22, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, xfp ** 3, thfp, xfp * thfp, xfp ** 2 * thfp, xfp ** 3 *thfp, thfp**2, xfp * thfp**2, xfp ** 2 * thfp, xfp ** 3 *thfp**2, thfp**3, xfp * thfp**3, xfp ** 2 * thfp**3, xfp ** 3 *thfp**3))
a33, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, xfp ** 3, xfp ** 4, thfp, xfp * thfp, xfp ** 2 * thfp, xfp ** 3 * thfp, xfp ** 4 *thfp, thfp**2, xfp * thfp**2, xfp ** 2 * thfp**2, xfp ** 3 * thfp**2, xfp ** 4 *thfp**2, thfp**3, xfp * thfp**3, xfp ** 2 * thfp**3, xfp ** 3 * thfp**3, xfp ** 4 *thfp**3, thfp**4, xfp * thfp**4, xfp ** 2 * thfp**4, xfp ** 3 * thfp**4, xfp ** 4 *thfp**4))
a44, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)

#PLOTING
x = np.linspace(min(xfp), max(xfp), 50)
theta = np.linspace(min(thfp), max(thfp), 50)
x, theta = np.meshgrid(x, theta)

r11 = N1_M1(x, theta, a11) # določiti a_{N,M}
chi11 = (N1_M1(xfp, thfp, a11) - thtg)**2 / (len(thtg) - len(a11))
r22 = N2_M2(x, theta, a22) # določiti a_{N,M}
chi22 = (N2_M2(xfp, thfp, a22) - thtg)**2 / (len(thtg) - len(a22))
r33 = N3_M3(x, theta, a33) # določiti a_{N,M}
chi33 = (N3_M3(xfp, thfp, a33) - thtg)**2 / (len(thtg) - len(a33))
r44 = N4_M4(x, theta, a44) # določiti a_{N,M}
chi44 = (N4_M4(xfp, thfp, a44) - thtg)**2 / (len(thtg) - len(a44))

r12 = N1_M2(x, theta, a12) # določiti a_{N,M}
chi12 = (N1_M2(xfp, thfp, a12) - thtg)**2 / (len(thtg) - len(a12))
r13 = N1_M3(x, theta, a13) # določiti a_{N,M}
chi13 = (N1_M3(xfp, thfp, a13) - thtg)**2 / (len(thtg) - len(a13))
r14 = N1_M4(x, theta, a14) # določiti a_{N,M}
chi14 = (N1_M4(xfp, thfp, a14) - thtg)**2 / (len(thtg) - len(a14))

r21 = N2_M1(x, theta, a21) # določiti a_{N,M}
chi21 = (N2_M1(xfp, thfp, a21) - thtg)**2 / (len(thtg) - len(a21))
r31 = N3_M1(x, theta, a31) # določiti a_{N,M}
chi31 = (N3_M1(xfp, thfp, a31) - thtg)**2 / (len(thtg) - len(a31))
r41 = N4_M1(x, theta, a41) # določiti a_{N,M}
chi41 = (N4_M1(xfp, thfp, a41) - thtg)**2 / (len(thtg) - len(a41))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s11 = ax.plot_surface(x, theta, r11, alpha=1, label='$N = 1, M = 1$')
s11._facecolors2d = s11._facecolor3d

s22 = ax.plot_surface(x, theta, r22, alpha=1, label='$N = 2, M = 2$')
s22._facecolors2d = s22._facecolor3d
s33 = ax.plot_surface(x, theta, r33, alpha=1, label='$N = 3, M = 3$') #slab fit
s33._facecolors2d = s33._facecolor3d
s44 = ax.plot_surface(x, theta, r44, alpha=1, label='$N = 4, M = 4$')
s44._facecolors2d = s44._facecolor3d

s12 = ax.plot_surface(x, theta, r12, alpha=1, label='$N = 1, M = 2$')
s12._facecolors2d = s12._facecolor3d
s13 = ax.plot_surface(x, theta, r13, alpha=1, label='$N = 1, M = 3$')
s13._facecolors2d = s13._facecolor3d
s14 = ax.plot_surface(x, theta, r14, alpha=1, label='$N = 1, M = 4$')
s14._facecolors2d = s14._facecolor3d

s21 = ax.plot_surface(x, theta, r21, alpha=1, label='$N = 2, M = 1$')
s21._facecolors2d = s21._facecolor3d
s31 = ax.plot_surface(x, theta, r31, alpha=1, label='$N = 3, M = 1$')
s31._facecolors2d = s31._facecolor3d
s41 = ax.plot_surface(x, theta, r41, alpha=1, label='$N = 4, M = 1$')
s41._facecolors2d = s41._facecolor3d

ax.scatter(xfp, thfp, thtg, color='black', label='podatki')
ax.set_xlabel('$x_{fp}$')
ax.set_ylabel('$\\theta_{fp}$')
ax.set_zlabel('$\\theta_{tg}$')
ax.grid(True)
plt.legend(loc='best')
plt.show()

#napake
fig2, ax2 = plt.subplots()
# ax2.plot(len(a11), sum(chi11), 'o', label='$N = M = 1$')

# ax2.plot(len(a22), sum(chi22), 'o', label='$N = M = 2$')
# ax2.plot(len(a33), sum(chi33), 'o', label='$N = M = 3$')
# ax2.plot(len(a44), sum(chi44), 'o', label='$N = M = 4$')

# ax2.plot(len(a12), sum(chi12), 'o', label='$M = 2$')
# ax2.plot(len(a13), sum(chi13), 'o', label='$M = 3$')
# ax2.plot(len(a14), sum(chi14), 'o', label='$M = 4$')

# ax2.plot(len(a21), sum(chi21), 'o', label='$N = 2$')
# ax2.plot(len(a31), sum(chi31), 'o', label='$N = 3$')
# ax2.plot(len(a41), sum(chi41), 'o', label='$N = 4$')

ax2.plot([len(a11), len(a22), len(a33), len(a44)], [sum(chi11), sum(chi22), sum(chi33), sum(chi44)], 'o-', label='$N = M$')
ax2.plot([len(a11), len(a12), len(a13), len(a14)], [sum(chi11), sum(chi12), sum(chi13), sum(chi14)], 'o-', label='$N = 1$')
ax2.plot([len(a11), len(a21), len(a31), len(a41)], [sum(chi11), sum(chi21), sum(chi31), sum(chi41)], 'o-', label='$M = 1$')

ax2.legend()
ax2.set_yscale('log')
ax2.set_xlabel('število parametrov')
ax2.set_ylabel('$\chi_r^2$')
plt.show()

print(sum(chi11), sum(chi12), sum(chi31))

# varčnejši model
def model_2(x_fp, theta_fp, a):
    a0,a1,a2,a3 = a
    return (a0+a1*x_fp+a2*x_fp**2) + (a3)*theta_fp
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, thfp)) #model2
a, residuals, rank, singular_values = np.linalg.lstsq(design_matrix, thtg, rcond=None)
res = N1_M1(x, theta, a) # določiti a_{N,M}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s11 = ax.plot_surface(x, theta, res, alpha=1)
s11._facecolors2d = s11._facecolor3d
ax.scatter(xfp, thfp, thtg, color='black', label='podatki')
ax.set_xlabel('$x_{fp}$')
ax.set_ylabel('$\\theta_{fp}$')
ax.set_zlabel('$\\theta_{tg}$')
ax.grid(True)
#plt.legend(loc='best')
plt.show()

exit()

################################################################################################################

def theta_tg_1(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5 = a
    return (a0+a1*x_fp+a2*x_fp**2) + (a3+a4*x_fp+a5*x_fp**2)*theta_fp
def theta_tg_2(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a8 = a
    return (a0+a1*x_fp+a2*x_fp**2) + (a3+a4*x_fp+a5*x_fp**2)*theta_fp + (a6+a7*x_fp+a8*x_fp**2)*theta_fp**2
def theta_tg_3(x_fp, theta_fp, a):
    a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11 = a
    return (a0+a1*x_fp+a2*x_fp**2) + (a3+a4*x_fp+a5*x_fp**2)*theta_fp + (a6+a7*x_fp+a8*x_fp**2)*theta_fp**2 + (a9+a10*x_fp+a11*x_fp**2)*theta_fp**3

data = np.loadtxt("C:/Users/ravni/Documents/FS/Modelska analiza I/thtg-xfp-thfp.dat")
thtg = []
xfp = []
thfp = []
for vrstica in data:
    thtg.append(vrstica[0])
    xfp.append(vrstica[1])
    thfp.append(vrstica[2])
thtg = np.array(thtg)
xfp = np.array(xfp)
thfp = np.array(thfp)

X = np.column_stack((xfp, thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, thfp, xfp * thfp, xfp ** 2 * thfp))
design_matrix = np.column_stack((np.ones_like(xfp), xfp, xfp ** 2, thfp, xfp * thfp, xfp ** 2 * thfp, 
                                 thfp ** 2, xfp * thfp ** 2, xfp ** 2 * thfp ** 2, 
                                 thfp ** 3, xfp * thfp ** 3, xfp ** 2 * thfp ** 3
                                 ))
a, _, _, _ = np.linalg.lstsq(design_matrix, thtg, rcond=None)
#U, S, Vt = np.linalg.svd(design_matrix, full_matrices=False)
#S_inv = np.diag(1/S)
#pseudo_inverse = Vt.T.dot(S_inv).dot(U.T)
#a = pseudo_inverse.dot(thtg)

#x_fp_grid, theta_fp_grid = np.meshgrid(xfp, thfp)
#approximated_values_3d = theta_tg_1(x_fp_grid.flatten(), theta_fp_grid.flatten(), a)
#approximated_values_3d = approximated_values_3d.reshape(x_fp_grid.shape)
x = np.linspace(min(xfp), max(xfp), 100)
theta = np.linspace(min(thfp), max(thfp), 100)
x, theta = np.meshgrid(x, theta)
rez = theta_tg_3(x, theta, a)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, theta, rez, cmap='viridis')
#ax.scatter(xfp, thfp, theta_tg_1(xfp,thfp,a))
ax.scatter(xfp, thfp, thtg, color='black')
ax.set_xlabel('$x_{fp}$')
ax.set_ylabel('$\\theta_{fp}$')
ax.set_zlabel('$\\theta_{tg}$')
ax.grid(True)
plt.show()