import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def hour_to_sec(i):
    h, m, s = i.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)
def GPS(i):
    x, y = map(float, i.split())
    return x, y
data = pd.read_csv('/mnt/nvme0n1p3/Users/ravni/Documents/FS/Modelska analiza I/podatki/DBF22-test-1.csv', skiprows=0, converters={
    0: lambda i: 0,  # Column 0 not useful
    1: hour_to_sec,  # Column 1
    5: GPS  # Column 5
})
t=data['Time'].values #s
t=t-t[0]
ground_speed=data['GSpd(kts)'].values* 	0.514444 #knots -> m/s
loc=data['GPS'].values #deg?
px=[loc[i][0] for i in range(len(loc))]
py=[loc[i][1] for i in range(len(loc))]
altitude=data['GAlt(m)'].values #m

i=236
t1=[t[i]]
gr_spd=[ground_speed[i]]
x=[px[i]]
y=[py[i]]
alt=[altitude[i]]
i=237
while ground_speed[i] > 0:
    t1.append(t[i])
    x.append(px[i])
    y.append(py[i])
    alt.append(altitude[i])
    gr_spd.append(ground_speed[i])
    i+=1
t1=np.array(t1-t1[0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc=ax.scatter(x, y, alt, c=t1, cmap='viridis', marker='o')
#norm = plt.Normalize(altitude.min(), altitude.max())
cbar = fig.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('čas [s]')
plt.grid(None)
plt.axis('on')
plt.xticks([])
plt.yticks([])
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('višina [m]')
plt.title('lege letenja')
plt.tight_layout()
plt.show()