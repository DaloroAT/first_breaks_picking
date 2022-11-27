import torch
from torch import nn
from torchvision import models


class Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_module = nn.Sequential[nn.Linear(1, 10), nn.Sigmoid()]
        self.z_module = nn.Sequential[nn.Linear(1, 10), nn.Sigmoid()]
        self.s_module = nn.Sequential[nn.Conv2d(1, 1, 1), nn.AdaptiveAvgPool2d((1, 1))]
        self.t = nn.Sequential[nn.Linear(3, 10), nn.Sigmoid(), nn.Linear(10, 1)]

    def forward(self, x, z, s):
        x = self.x_module(x)
        z = self.t_module(z)
        s = self.s_module(s)
        return self.t(torch.cat([x, z, s], dim=1))


class TravelTimeModule(nn.Module):
    def __init__(self, slowness):
        super().__init__()
        self.slowness = nn.Parameter(slowness.float(), requires_grad=True)
        self.solver = Solver()

    def net_t(self, x, z):
        return self.solver(x=x, z=z, s=self.slowness)

    def net_residual(self, x, z):
        t = self.net_t(x=x, z=z)

        t_x = torch.autograd.grad(
            t, x,
            grad_outputs=torch.ones_like(t),
            retain_graph=True,
            create_graph=True
        )[0]

        t_z = torch.autograd.grad(
            t, z,
            grad_outputs=torch.ones_like(t),
            retain_graph=True,
            create_graph=True
        )[0]

        return t_x ** 2 + t_z ** 2 - self.slowness ** 2

    def loss(self, x, z, t):
        t_pred = self.net_t(x, z)
        res = self.net_residual(x, z)
        return torch.mean((t - t_pred) ** 2) + torch.mean(res ** 2)


