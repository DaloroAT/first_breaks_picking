import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import tqdm

from first_breaks._pytorch.tomo.velocity import VelocityVerticalLayers, VelocityGrid, get_grid_based_points

# X_GRID = [1, 2, 3, 4, 5]
# Z_GRID = [
#     1,
#     2,
#     3,
#     4,
#     # 5
# ]
# V_GRID = [
#     [1, 2, 3, 4, 5],
#     [5, 4, 3, 2, 1],
#     [4, 5, 6, 6, 7],
#     [4, 5, 6, 7, 8],
#     # [6, 5, 4, 3, 2]
# ]
X_GRID = torch.linspace(1, 5, 30)
Z_GRID = torch.linspace(1, 5, 40)
V_GRID = torch.linspace(1, 2, len(Z_GRID))[:, None] + torch.linspace(1, 3, len(X_GRID))[None, :]

target_velocity_model = VelocityGrid(vel_grid=V_GRID,
                                     x_grid=X_GRID,
                                     z_grid=Z_GRID,
                                     x_smoothing=0.001,
                                     z_smoothing=0.001,
                                     )
# target_velocity_model.plot(x_min=0, x_max=max(X_GRID), z_min=0, z_max=max(Z_GRID), nx=100, nz=100)
# print(target_velocity_model.vel_model)
# print(target_velocity_model.z_model)
# print(target_velocity_model.x_model)

# assert False

# num_points = 200
# depth = torch.linspace(0, 15, num_points)
# point = torch.zeros((num_points, 2))
# point[:, -1] = depth


point = get_grid_based_points(x_min=0, x_max=max(X_GRID), nx=100, z_min=0, z_max=max(Z_GRID), nz=100)


with torch.no_grad():
    target_v = target_velocity_model(point)


model = VelocityGrid(vel_grid=torch.ones_like(V_GRID),
                     x_grid=X_GRID,
                     z_grid=Z_GRID,
                     x_smoothing=0.001,
                     z_smoothing=0.001,
                     learnable_cell_size=False,
                     use_weighted_forward=False)
optimizer = Adam(model.parameters(), lr=1e-1)


pbar = tqdm(range(100))

for _ in pbar:
    optimizer.zero_grad()

    point_train = point.detach().clone().requires_grad_(True)
    pred_v = model(point_train)

    loss = mse_loss(pred_v, target_v)

    loss.backward()
    optimizer.step()

    pbar.set_postfix({"loss": loss.item(), "min": pred_v.min().item(), "max": pred_v.max().item()})


target_velocity_model.plot(x_min=0, x_max=max(X_GRID), z_min=0, z_max=max(Z_GRID), nx=100, nz=100,
                           v_min=0.7, v_max=8.5, title='Target')
model.plot(x_min=0, x_max=max(X_GRID), z_min=0, z_max=max(Z_GRID), nx=100, nz=100,
           v_min=0.7, v_max=8.5, title='Model')

# print(model.vel_model)
# print(model.z_model)
# print(model.x_model)

