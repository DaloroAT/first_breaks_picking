import warnings

import matplotlib.pyplot as plt
import torch
from torch import nn


class VelocityBase(nn.Module):
    @torch.no_grad()
    def plot_vertical(self, z_min, z_max, n=1000, v_min=None, v_max=None, title=None):
        points = torch.zeros(n, 2)
        points[:, -1] = torch.linspace(z_min, z_max, len(points))
        v_forward = self(points)

        plt.plot(points[:, -1], v_forward.squeeze())
        if title is None:
            title = f'Model {self.__class__.__name__}'
        plt.title(title)
        plt.xlabel('depth')
        plt.ylabel('velocity')
        if v_min or v_max:
            v_min = v_forward.min().item() if v_min is None else v_min
            v_max = v_forward.max().item() if v_max is None else v_max
            plt.ylim([v_min, v_max])
        plt.show()


class VelocityVerticalLayers(VelocityBase):
    def __init__(self, vel, depth, smoothing=None):
        super(VelocityVerticalLayers, self).__init__()
        assert len(vel) == len(depth) > 1

        vel = torch.tensor(vel, dtype=torch.float)
        # we add extra value to vectorize forward
        depth = torch.cat([torch.tensor([-1.0]), torch.cumsum(torch.tensor(depth, dtype=torch.float), dim=0)])

        self.vel_model = nn.Parameter(vel)
        self.depth_model = nn.Parameter(depth)

        if smoothing is None:
            smoothing = (depth[1:] - depth[0:-1]).min() / 50
        self.smoothing = smoothing
        print(f"{self.smoothing=}")

    def forward(self, point):
        x = point[:, -1]
        # Use None to add an extra dimension to x for broadcasting
        x = x[:, None]

        # Compute the value of the step functions
        up_steps = torch.sigmoid((x - self.depth_model[:-1]) / self.smoothing)
        down_steps = 1 - torch.sigmoid((x - self.depth_model[1:]) / self.smoothing)

        y = torch.sum(self.vel_model * up_steps * down_steps, dim=1)

        return y


class VelocityGrid(nn.Module):
    def __init__(self,
                 vel_grid,
                 x_grid,
                 z_grid,
                 x_smoothing=None,
                 z_smoothing=None,
                 learnable_cell_size=False,
                 use_weighted_forward=False):
        super().__init__()
        if isinstance(vel_grid, torch.Tensor):
            vel_grid = vel_grid.clone().detach()
        else:
            vel_grid = torch.tensor(vel_grid, dtype=torch.float32)
        if isinstance(x_grid, torch.Tensor):
            x_grid = x_grid.clone().detach()
        else:
            x_grid = torch.tensor(x_grid, dtype=torch.float32)
        if isinstance(z_grid, torch.Tensor):
            z_grid = z_grid.clone().detach()
        else:
            z_grid = torch.tensor(z_grid, dtype=torch.float32)

        for array in [x_grid, z_grid]:
            assert array.ndim == 1
            assert (array >= 0).all()
            assert torch.equal(torch.sort(array, descending=False)[0], array)
            assert len(torch.unique(array)) == len(array)

        assert vel_grid.ndim == 2
        assert vel_grid.shape == (len(z_grid), len(x_grid)), (f"Velocity shape {vel_grid.shape}, "
                                                              f"spatial grid shape {(len(z_grid), len(x_grid))}")

        if learnable_cell_size and not use_weighted_forward:
            raise RuntimeError("For cell size learning, flag use_weighted_forward has to be `True`, but got `False`. "
                               "Please set `use_weighted_forward=True` or disable cell size learning.")

        if not learnable_cell_size and use_weighted_forward:
            warnings.warn("You should consider to set `use_weighted_forward=False` to increase performance if you "
                          "don't perform cell size learning")

        self.vel_model = nn.Parameter(vel_grid)
        # we add extra point -1 outside the model to vectorize forward pass
        self.x_model = torch.cat([torch.tensor([-1.]), x_grid])
        self.z_model = torch.cat([torch.tensor([-1.]), z_grid])
        if learnable_cell_size:
            self.x_model = nn.Parameter(self.x_model)
            self.z_model = nn.Parameter(self.z_model)

        if x_smoothing is None:
            x_smoothing = (x_grid[1:] - x_grid[0:-1]).min() / 50
        self.x_smoothing = x_smoothing

        if z_smoothing is None:
            z_smoothing = (z_grid[1:] - z_grid[0:-1]).min() / 50
        self.z_smoothing = z_smoothing

        print(f"{self.x_smoothing=}, {self.z_smoothing=}")

        self.use_weighted_forward = use_weighted_forward

    def forward(self, point):
        x = point[:, 0].view(-1, 1)
        z = point[:, 1].view(-1, 1)

        x_weights = torch.sigmoid((x - self.x_model[:-1]) / self.x_smoothing) * \
                    (1 - torch.sigmoid((x - self.x_model[1:]) / self.x_smoothing))
        z_weights = torch.sigmoid((z - self.z_model[:-1]) / self.z_smoothing) * \
                    (1 - torch.sigmoid((z - self.z_model[1:]) / self.z_smoothing))

        if self.use_weighted_forward:
            x_weights = x_weights / x_weights.sum(dim=-1, keepdim=True)
            z_weights = z_weights / z_weights.sum(dim=-1, keepdim=True)

            vel = (self.vel_model[None, :, :] * z_weights[..., None] * x_weights[..., None, :]).sum(dim=(1, 2))
        else:
            indices_x = torch.argmax(x_weights, dim=1)
            indices_z = torch.argmax(z_weights, dim=1)

            # Gather values from 2D tensor using indices
            vel = self.vel_model[indices_z, indices_x]

        return vel.view(-1, 1)
