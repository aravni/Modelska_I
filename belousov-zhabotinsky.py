import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation

# Width, height of the image.
nx, ny = 500, 500
# Reaction parameters.
alpha, beta, gamma = 1,0.1,0.1

def update(p,arr):
    """Update arr[p] to arr[q] by evolving in time."""

    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
    q = (p+1) % 2
    s = np.zeros((3, ny,nx))
    m = np.ones((3,3)) / 9
    for k in range(3):
        s[k] = convolve2d(arr[p,k], m, mode='same', boundary='wrap')
    # Apply the reaction equations
    arr[q,0] = s[0] + s[0]*(alpha*s[1] - gamma*s[2])
    arr[q,1] = s[1] + s[1]*(beta*s[2] - alpha*s[0])
    arr[q,2] = s[2] + s[2]*(gamma*s[0] - beta*s[1])
    # Ensure the species concentrations are kept within [0,1].
    np.clip(arr[q], 0, 1, arr[q])
    return arr

# Initialize the array with random amounts of A, B and C.
arr = np.random.random(size=(2, 3, ny, nx))
# Initialize the array with 10 near central points of substrate.
#arr = np.zeros((2, 3, ny, nx))
#center_x, center_y = nx // 2, ny // 2
#for i in range(-5, 6):
#    for j in range(-5, 6):
#        arr[0, 0, center_y + i, center_x + j] = 0.5  # Set the central points with substrate concentration to 1.0

# Initialize the array with the same concentration for all points.
concentration_value = 0.5  # Set the desired concentration value
arr = np.full((2, 3, ny, nx), concentration_value, dtype=float)

# Set up the image
fig, ax = plt.subplots()
im = ax.imshow(arr[0,0], cmap=plt.cm.winter)
ax.axis('off')

def animate(i, arr):
    """Update the image for iteration i of the Matplotlib animation."""

    arr = update(i % 2, arr)
    im.set_array(arr[i % 2, 0])
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=200, interval=5,
                               blit=False, fargs=(arr,))

# To view the animation, uncomment this line
plt.show()

# To save the animation as an MP4 movie, uncomment this line
#anim.save(filename='bz.mp4', fps=30)
