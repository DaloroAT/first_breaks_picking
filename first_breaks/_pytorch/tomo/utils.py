import torch
import matplotlib.pyplot as plt
from torch import nn


@torch.no_grad()
def plot_predict_on_grid(model: nn.Module, z_min, z_max, x_min, x_max, nx=100, nz=100, v_min=None, v_max=None, title=None):
    points = get_grid_based_points(z_min=z_min, z_max=z_max, x_min=x_min, x_max=x_max, nx=nx, nz=nz)

    # prev_state = model.training
    # model.eval()
    vel = model(points)
    # model.train(prev_state)

    if v_min:
        vel[vel < v_min] = v_min

    if v_max:
        vel[vel > v_max] = v_max

    vel_grid = vel.cpu().numpy()
    vel_grid = vel_grid.reshape((nz, nx), order='F')

    plt.imshow(vel_grid, extent=[x_min, x_max, z_max, z_min])
    plt.colorbar()
    if title is None:
        title = f'Model {model.__class__.__name__}'
    plt.title(title)
    plt.show()


def get_grid_based_points(z_min, z_max, x_min, x_max, nx=100, nz=100):
    x_vec = torch.linspace(x_min, x_max, nx)
    z_vec = torch.linspace(z_min, z_max, nz)
    xx, zz = torch.meshgrid([x_vec, z_vec])
    xx = xx.flatten().view(-1, 1)
    zz = zz.flatten().view(-1, 1)

    return torch.cat([xx, zz], dim=1)
