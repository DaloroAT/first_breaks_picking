import torch
from torch import nn
import matplotlib.pyplot as plt


class VelocityBase(nn.Module):
    @torch.no_grad()
    def plot_vertical(self, z_min, z_max, n=1000):
        points = torch.zeros(n, 2)
        points[:, -1] = torch.linspace(z_min, z_max, len(points))
        v_forward = self(points)

        plt.plot(points[:, -1], v_forward.squeeze())
        plt.title(f'Model {self.__class__.__name__}')
        plt.xlabel('depth')
        plt.ylabel('velocity')
        plt.show()


class VelocityConst(VelocityBase):
    def __init__(self, v):
        super(VelocityConst, self).__init__()
        self.model = nn.Parameter(torch.tensor([v], dtype=torch.float), requires_grad=False)

    @torch.no_grad()
    def forward(self, point):
        outp = self.model.repeat(len(point)).view(-1, 1)
        return outp


class VelocityVerticalLayers(VelocityBase):
    def __init__(self, vel, depth, smoothing=None):
        super(VelocityVerticalLayers, self).__init__()
        assert len(vel) == len(depth) > 1
        self.vel = vel
        self.depth = depth

        vel = torch.tensor(vel, dtype=torch.float)
        depth = torch.cumsum(torch.tensor(depth, dtype=torch.float), dim=0)

        self.vel_model = nn.Parameter(vel, requires_grad=False)
        self.depth_model = nn.Parameter(depth, requires_grad=False)

        if smoothing is None:
            smoothing = (depth[1:] - depth[0:-1]).min() / 50
        self.smoothing = smoothing

    @torch.no_grad()
    def forward(self, point):
        z = point[:, -1]
        output = []
        for i, (z0_i, z1_i, v_i) in enumerate(zip([0, *self.depth_model[0:-1]], self.depth_model, self.vel_model)):
            if i == 0:
                step = torch.sigmoid((z1_i - z) / self.smoothing)
            elif i == len(self.vel_model) - 1:
                step = torch.sigmoid((z - z0_i) / self.smoothing)
            else:
                step = torch.sigmoid((z - z0_i) / self.smoothing) - torch.sigmoid((z - z1_i) / self.smoothing)
            output_i = v_i * step
            output.append(output_i)
        output = sum(output).view(-1, 1)
        return output


class VelocityVerticalGrad(VelocityBase):
    def __init__(self, vel, depth):
        super(VelocityVerticalGrad, self).__init__()
        assert len(vel) == len(depth) == 2
        assert depth[1] > depth[0]
        self.vel = vel
        self.depth = depth

        vel = torch.tensor(vel, dtype=torch.float)
        depth = torch.tensor(depth, dtype=torch.float)

        self.vel_model = nn.Parameter(vel, requires_grad=False)
        self.depth_model = nn.Parameter(depth, requires_grad=False)

    @torch.no_grad()
    def forward(self, point):
        z = point[:, -1]
        k = (self.vel_model[1] - self.vel_model[0]) / (self.depth_model[1] - self.depth_model[0])
        b = self.vel_model[0] - k * self.depth_model[0]
        v = (k * z + b).view(-1, 1)
        assert torch.all(v > 0)
        return v
