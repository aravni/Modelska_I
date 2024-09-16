import numpy as np
import matplotlib.pyplot as plt
import statistics

def z(x,y):
    return 3+2*x-x**2+y**2-4*x*y
def z2(x,y, a):
    a1,a2,a3,a4,a5 = a
    return a1+a2*x-a3*x**2+a4*y**2-a5*x*y
def z3(x,y,a):
    a0,a1,a2 = a
    return a0+a1*x**2-a2*y**2

x0 = np.linspace(-10,10,10)
y0 = np.linspace(-10,10,10)
x, y = np.meshgrid(x0, y0)
#z_val = z(x, y)
#z_val_o = z2(x,y, a=[0,2,3,4,5])
z_val_o = z3(x,y, [0,5,5])
z_val = np.array([i[j] + 0 * np.random.uniform(-1,1) for i in z_val_o for j in range(len(z_val_o[0]))]) #šum

#design_matrix = np.column_stack((np.ones_like(x.flatten()), x.flatten(), -x.flatten() ** 2, y.flatten() ** 2, -x.flatten() * y.flatten()))
#a, _, _, _ = np.linalg.lstsq(design_matrix, z_val.flatten(), rcond=None)
design_matrix = np.column_stack((np.ones_like(x.flatten()),  x.flatten() ** 2, -y.flatten() ** 2))
a, _, _, _ = np.linalg.lstsq(design_matrix, z_val.flatten(), rcond=None)
print('a_0 = ', a[0], ', a_1 = ', a[1], ', a_2 = ', a[2])

#z_2_val = z2(x, y, a)
z_2_val = z3(x,y,a)
chi = (z_2_val.flatten() - z_val.flatten())**2 / (len(z_val.flatten())-len(a)) /3 #reduciran chi^2
print('chi = ', sum(chi))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z_val)
surf = ax.plot_surface(x, y, z_2_val, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()

# variacije šuma
x0 = np.linspace(-10,10,100)
y0 = np.linspace(-10,10,100)
x, y = np.meshgrid(x0, y0)
z_val_o = z3(x,y, [0,5,5])

chi_data = []
šum = []
for val in range(0, 101, 5):
    z_val = np.array([i[j] + val * np.random.uniform(-1,1) for i in z_val_o for j in range(len(z_val_o[0]))]) #šum
    design_matrix = np.column_stack((np.ones_like(x.flatten()),  x.flatten() ** 2, -y.flatten() ** 2))
    a, _, _, _ = np.linalg.lstsq(design_matrix, z_val.flatten(), rcond=None)
    #print(val)
    #print('a_0 = ', a[0], ', a_1 = ', a[1], ', a_2 = ', a[2])
    z_2_val = z3(x,y,a)
    chi = (z_2_val.flatten() - z_val.flatten())**2 / (len(z_val.flatten())-len(a)) /3
    #print('chi = ', sum(chi))
    chi_data.append(sum(chi))
    šum.append(val)

plt.plot(šum, chi_data, '.-')
plt.xlabel('šum')
plt.ylabel('$\chi_r^2$')
plt.show()