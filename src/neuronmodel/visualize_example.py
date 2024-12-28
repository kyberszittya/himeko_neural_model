import numpy as np
import plotly.graph_objects as go

# Create volumetric data (e.g., 10x10x10)
x, y, z = np.mgrid[0:10, 0:10, 0:10]
values = np.sin(x**2 + y**2 + z**2)

fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=values.flatten(),
    isomin=-1,
    isomax=1,
    opacity=0.1,  # Adjust transparency
    surface_count=17,  # Number of contours
))

fig.show()