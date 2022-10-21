import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)


PI = torch.tensor(torch.pi)
PI_M2 = 2 * PI
PI_M3_D2 = 3 / 2 * PI
ZERO = torch.tensor(0)

RIGHT = torch.tensor(1)
TOP = torch.tensor(2)
LEFT = torch.tensor(3)
BOTTOM = torch.tensor(4)


def calc_borders_angles(x0, z0, h, w) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    right_bottom = torch.arctan(torch.divide(w - x0, h - z0))
    right_top = PI - torch.arctan(torch.divide(w - x0, z0))
    left_top = PI_M3_D2 - torch.arctan(torch.divide(z0, x0))
    left_bottom = PI_M2 - torch.arctan(torch.divide(x0, h - z0))
    return right_bottom, right_top, left_top, left_bottom


def calc_x(z, x0, z0, theta) -> torch.Tensor:
    return x0 + (z - z0) * torch.tan(theta)


def calc_z(x, x0, z0, theta) -> torch.Tensor:
    return z0 + (x - x0) / torch.tan(theta)


def find_next_points_and_border(x0, z0, h, w, theta) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = len(x0)
    right_bottom, right_top, left_top, left_bottom = calc_borders_angles(x0, z0, h, w)

    border = torch.empty(num_points)
    x = torch.empty(num_points)
    z = torch.empty(num_points)

    right_mask = torch.logical_and(right_bottom < theta, theta <= right_top)
    border[right_mask] = RIGHT
    x[right_mask] = w
    z[right_mask] = calc_z(x[right_mask], x0[right_mask], z0[right_mask], theta[right_mask])

    top_mask = torch.logical_and(right_top < theta, theta <= left_top)
    border[top_mask] = TOP
    z[top_mask] = ZERO
    x[top_mask] = calc_x(z[top_mask], x0[top_mask], z0[top_mask], theta[top_mask])

    left_mask = torch.logical_and(left_top < theta, theta <= left_bottom)
    border[left_mask] = LEFT
    x[left_mask] = ZERO
    z[left_mask] = calc_z(x[left_mask], x0[left_mask], z0[left_mask], theta[left_mask])

    bottom_mask = torch.logical_or(torch.logical_and(left_bottom < theta, theta <= PI_M2),
                                   torch.logical_and(0 <= theta, theta <= right_bottom))
    border[bottom_mask] = RIGHT
    z[bottom_mask] = h
    x[bottom_mask] = calc_x(z[bottom_mask], x0[bottom_mask], z0[bottom_mask], theta[bottom_mask])

    return x, z, border


def calc_next_init_border_and_angles(theta, border):
    pass


N_RAYS = 500
NX = 10
NZ = 10
VELOCITY = 1000 + 20 * torch.randn(NZ, NX)
H = torch.tensor(2)
W = torch.tensor(2)
X0 = torch.linspace(0, W, N_RAYS)
Z0 = torch.linspace(0, H, N_RAYS)
THETA = torch.linspace(0, PI_M2, N_RAYS)

# print(torch.asin(torch.tensor(0.9)).isnan())
#
# print(360 - torch.tensor(torch.nan))

# st = time.perf_counter()
# for _ in range(1):
#     find_next_points_and_border(X0, Z0, H, W, THETA)
# print(time.perf_counter() - st)


v1 = 1300
v2 = 1100
theta_1 = PI_M3_D2 + torch.deg2rad(torch.tensor(45))

theta_2 = PI_M2 - torch.asin(v2 / v1 * torch.sin(PI_M2 - theta_1))

print(torch.rad2deg(theta_1), torch.rad2deg(theta_2))


