import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import tqdm

from first_breaks._pytorch.tomo.velocity import VelocityVerticalLayers

V_LAYERS = [1, 2, 3, 1.25, 3]
H_LAYERS = [2, 3, 4, 2, 5]
NUM_LAYERS = len(H_LAYERS)

target_velocity_model = VelocityVerticalLayers(V_LAYERS, H_LAYERS)
# target_velocity_model.plot_vertical(0, 15)

# assert False

num_points = 200
depth = torch.linspace(0, 15, num_points)
point = torch.zeros((num_points, 2))
point[:, -1] = depth

with torch.no_grad():
    target_v = target_velocity_model(point)


model = VelocityVerticalLayers([1] * sum(H_LAYERS), [1] * sum(H_LAYERS), smoothing=0.04)
optimizer = Adam(model.parameters(), lr=1e-1)


pbar = tqdm(range(200))

for _ in pbar:
    optimizer.zero_grad()

    point_train = point.detach().clone().requires_grad_(True)
    pred_v = model(point_train)

    loss = mse_loss(pred_v, target_v)

    loss.backward()
    optimizer.step()

    pbar.set_postfix({"loss": loss.item(), "min": pred_v.min().item(), "max": pred_v.max().item()})


target_velocity_model.plot_vertical(0, 15, v_min=0.5, v_max=3.3, title='Target')
model.plot_vertical(0, 15, v_min=0.5, v_max=3.3, title='Model')

print(model.depth_model)
