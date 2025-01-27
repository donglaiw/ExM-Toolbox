import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_pts(src: np.ndarray, dst: np.ndarray, symbol1='circle', symbol2='diamond-open'):
    fig = go.Figure()

    # trace of source point cloud
    fig.add_trace(go.Scatter3d(
        x = src[:,0],
        y = src[:,1],
        z = src[:,2],
        mode = 'markers',
        marker = dict(
            size = 5,
            opacity = 0.5,
            symbol = symbol1,
            color = 'green'
        ),
        text = [str(x) for x in range(len(src))]
    ))

   # trace of target point cloud
   fig.add_trace(go.Scatter3d(
       x = dst[:,0],
       y = dst[:,1],
       z = dst[:,2],
       mode = 'markers',
       marker = dict(
           size = 5,
           opacity = 0.5,
           symbol = symbol2,
           color = 'red'
       ),
       text = [str(x) for x in range(len(dst))]
   ))

   # plot the figure
   fig.update_layout(
       autosize = False,
       width = 1000,
       height = 800
   )
   fig.show()

