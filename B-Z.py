import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Constants
k1 = 2
k2 = 3
g = 0.5
interval = 500  # Display interval

# Grid dimensions
dx = 500
x0 = 1
width, height = dx, dx
print('N = ', dx*dx)

# Initialize the grid with random values between 0 and 2
grid = np.random.randint(0, 3, (width, height), dtype=int)
#grid = np.zeros((width, height))
for i in range(100):
    loc_x, loc_y = np.random.random(2) * (width, height)
    #grid[int(loc_x - x0): int(loc_x + x0), int(loc_y - x0): int(loc_y + x0)] = np.random.randint(0,3)
for i in range(10):
    loc_x, loc_y = np.random.random(2) * (width, height)
    #grid[int(loc_x - x0): int(loc_x + x0), int(loc_y - x0): int(loc_y + x0)] = 2

# Define the Moore neighborhood
neighborhood = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

# Set up the image
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap=plt.cm.winter)
ax.axis('off')

def update_grid(grid):
    new_grid = grid.copy()
    for x in range(width):
        for y in range(height):
            cell = grid[x, y]
            neighbors = grid[x - 1:x + 2, y - 1:y + 2]
            a = np.sum(neighbors == 1)  # Count infected cells
            b = np.sum(neighbors == 2)  # Count ill cells

            if cell == 0:
                new_grid[x, y] = int(a / k1) + int(b / k2)
            elif cell == 2:
                new_grid[x, y] = 0
            else:
                s = np.sum(neighbors)
                new_grid[x, y] = int(s / (a + b + 1)) + g
    return new_grid

def animate(i):
    """Update the image for iteration i of the Matplotlib animation."""
    global grid
    grid = update_grid(grid)
    im.set_array(grid)
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=1, interval=interval,
                               blit=False)
plt.show()
#anim.save(filename='bz.mp4', fps=30)