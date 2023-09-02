import pandas as pd
import torch
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import tqdm

from first_breaks._pytorch.tomo.models import Tau, Eikonal, Model
from first_breaks._pytorch.tomo.utils import plot_predict_on_grid
from first_breaks._pytorch.tomo.velocity import VelocityGrid
from first_breaks.const import PROJECT_ROOT
import matplotlib.pyplot as plt


df = pd.read_csv(PROJECT_ROOT / "surf_forward_times_fixed.ST", skipinitialspace=True, delimiter=" ", index_col=False)
print(df["ft"].max(), df["ft"].min())
print(df["sx"].max(), df["sx"].min())
print(df["rx"].max(), df["rx"].min())

for col in ["sx", "sy", "sz", "rx", "ry", "rz", "ft"]:
    df[col] /= 1000

print(df["ft"].max(), df["ft"].min())
print(df["sx"].max(), df["sx"].min())
print(df["rx"].max(), df["rx"].min())


tau_model = Tau(input_dim=2, hidden_dim=20, num_layers=10, max_value=2)
x_grid = torch.linspace(0, 5, 200)
z_grid = torch.linspace(0, 1, 100)
# velocity_model = VelocityGrid(vel_grid=2.5 * torch.ones((len(z_grid), len(x_grid))),
#                               x_grid=x_grid,
#                               z_grid=z_grid,
#                               use_weighted_forward=True,
#                               learnable_cell_size=False)
velocity_model = Model(input_dim=2, hidden_dim=10, num_layers=10, max_value=3)

eikonal = Eikonal(tau_model=tau_model, background_velocity=None)

source_df = torch.cat([
    torch.tensor(df["sx"], dtype=torch.float32)[:, None],
    torch.tensor(df["sz"], dtype=torch.float32)[:, None]
],
    dim=1)
receiver_df = torch.cat([
    torch.tensor(df["rx"], dtype=torch.float32)[:, None],
    torch.tensor(df["rz"], dtype=torch.float32)[:, None]
], dim=1)

source_point = torch.cat([source_df, receiver_df], dim=0)
receiver_point = torch.cat([receiver_df, source_df], dim=0)

travel_time_target = torch.tensor(df["ft"], dtype=torch.float32)[:, None]
travel_time_target = torch.cat([travel_time_target, travel_time_target], dim=0)

print(travel_time_target.max(), travel_time_target.min())


# plt.hist(df["ft"])
# plt.show()
#
# assert False


lr_eikonal = 3e-2
lr_velocity = 3e-2
num_epochs = 200

optimizer_eikonal = Adam(list(eikonal.parameters()), lr_eikonal)
optimizer_velocity = Adam(list(velocity_model.parameters()), lr_velocity)

pbar = tqdm(range(num_epochs))

for _ in pbar:
    optimizer_eikonal.zero_grad()
    optimizer_velocity.zero_grad()

    s_train = source_point.clone().detach().requires_grad_(False)
    r_train = receiver_point.clone().detach().requires_grad_(True)

    # travel_time_pred = eikonal(s_train, r_train)

    travel_time_pred, velocity_eikonal_pred = eikonal.get_time_and_velocity(s_train, r_train)

    velocity_eikonal_pred = velocity_eikonal_pred.clone().detach().requires_grad_(False)
    velocity_model_pred = velocity_model(r_train).view(-1, 1)

    loss_data = mse_loss(travel_time_pred, travel_time_target)
    loss_velocity = mse_loss(velocity_model_pred, velocity_eikonal_pred)

    loss_data.backward()
    optimizer_eikonal.step()

    # loss_velocity.backward()
    # optimizer_velocity.step()

    # loss = loss_velocity + 2 * loss_data
    #
    # loss.backward()
    # optimizer.step()

    pbar.set_postfix({
        "loss_velocity": loss_velocity.item(),
        "loss_data": loss_data.item(),
        # "v_min": velocity_model.vel_model.min().item(),
        # "v_max": velocity_model.vel_model.max().item()
    })


# print(velocity_model.vel_model.min(), velocity_model.vel_model.max())

plot_predict_on_grid(velocity_model,
                     x_min=x_grid.min(), x_max=x_grid.max(), z_min=z_grid.min(), z_max=z_grid.max(),
                     nx=len(x_grid), nz=len(z_grid)
                     )

# velocity_model.plot(x_min=x_grid.min(), x_max=x_grid.max(), z_min=z_grid.min(), z_max=z_grid.max(),
#                     nx=len(x_grid), nz=len(z_grid))




