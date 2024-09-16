import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
#import dash_core_components as dcc
from dash import html
#import dash_html_components as html
from dash import html

# Constants
k1 = 2
k2 = 3
g = 0.1

# Grid dimensions
dx = 50
width, height = dx, dx

# Initialize the grid with random values between 0 and 2
grid = np.random.randint(0, 3, (width, height), dtype=int)
#grid = np.zeros((width, height))
x0 = 1
#grid[int(width/2-x0) : int(width/2+x0), int(height/2-x0) : int(height/2+x0)] = 1

# Define the Moore neighborhood
neighborhood = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])

# Initialize the Dash app
app = dash.Dash(__name__)

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
            elif cell == 1:  # Change this condition from cell == 2 to cell == 1
                new_grid[x, y] = 0
            else:
                s = np.sum(neighbors)
                new_grid[x, y] = int(s / (a + b + 1)) + g

    return new_grid

# Create the initial plot
def create_plot(grid):
    fig = go.Figure(data=[go.Heatmap(z=grid, colorscale='Viridis', zmin=0, zmax=2)])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(width=width, height=height)
    return fig

fig = create_plot(grid)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(figure=fig, id='bz-reaction-plot'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0
    )
])

# Define callback to update the plot
@app.callback(
    Output('bz-reaction-plot', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_plot(n_intervals):
    global grid
    grid = update_grid(grid)
    return create_plot(grid)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)