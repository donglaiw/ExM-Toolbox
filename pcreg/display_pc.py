import math
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

'''
Plot 200 random points (fixed cloud) and their
nearest neighbors in the moving cloud
'''

def find_nn(n, X_fix, X_mov):
    """
    args:   n     -> number of random points
            X_fix -> fixed cloud coordinates
            X_mov -> moving cloud coordinates

    returns the coordinates of the nearest neighbors
    """
    nn_idx = []
    X_mov_nn = np.zeros((n, 3))
    idx = np.random.choice(X_fix.shape[0], n, replace=False)

    for i in range(n):
        X_fix_rand[i] = X_fix[idx[i]]

    # find index of nearest neighbors
    for i in range(n):
        dist_list = []
        pt1 = X_fix_rand[i]
        min_dist = math.inf

        for j in range(n):
            pt2 = X_mov[j]
            dist = np.sqrt(np.sum(np.square(pt2-pt1)))
            dist_list.append(dist)

        for idx in range(n):
            if(dist_list[idx] == min(dist_list)):
                nn_idx.append(idx)

    # get nearest neighbor coordinates
    for i in range(n):
        X_mov_nn[i] = X_mov[j]


'''
Plot points
'''

def plot_overlap(X_fix_rand, X_mov_nn, X_reg_nn):
    '''
    plot all clouds together
    '''
    fix_data = pd.DataFrame(X_fix_rand, columns=['x', 'y', 'z'])
    fix_data['point_cloud'] = 'Fixed cloud'

    moved_data = pd.DataFrame(X_mov_nn, columns=['x', 'y', 'z'])
    moved_data['point_cloud'] = 'Moving cloud'

    registered_data = pd.DataFrame(X_reg_nn, columns=['x', 'y', 'z'])
    registered_data['point_cloud'] = 'Registered Cloud'

    frames = [fix_data, moved_data, registered_data]
    pc_data = pd.concat(frames)

    fig = px.scatter_3d(pc_data, x='x', y='y', z='z',
                        opacity=0.5, color='point_cloud')
    fig.show()


# plot n pts (otherwise there's too much clutter)
def plot_adjacent(X_fix, X_mov_, n):
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "scene"}, {"type": "scene"}]],)

    fig.add_trace(go.Scatter3d(x=X_fix[0:n:,0], y=X_fix[0:n:,1],
                           z=X_fix[0:n:, 2], mode="markers",
                               opacity=0.5),
                  row=1, col=1)
    fig.add_trace(go.Scatter3d(x=X_mov[0:n:,0], y=X_mov[0:n:,1],
                           z=X_mov[0:n:, 2], mode="markers",
                               opacity=0.5),
                  row=1, col=2)

