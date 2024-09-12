import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from .viewer_engine import gl

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, m, centroid


def get_points_mesh(points, size, colors = None):

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if colors is None:
        # feature = normalize_feature(points[:,2])
        feature = points[:,2]
        norm = mpl.colors.Normalize(vmin=-2.5, vmax=1.5)
        # norm = mpl.colors.Normalize(vmin=feature.min()+0.5, vmax=feature.max()-0.5)
        cmap = cm.jet 
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5

    else:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()

    mesh = gl.GLScatterPlotItem(pos=np.asarray(points[:, 0:3]), size=size, color=colors)

    return mesh