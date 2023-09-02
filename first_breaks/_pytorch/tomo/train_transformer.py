import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import mse_loss, l1_loss, softmax, sigmoid
from torch.optim import Adam
from tqdm import tqdm

from first_breaks._pytorch.tomo.models import Tau, Eikonal, Model
from first_breaks._pytorch.tomo.transformer import TomoTransformer
from first_breaks._pytorch.tomo.utils import plot_predict_on_grid, get_grid_based_points
from first_breaks._pytorch.tomo.velocity import VelocityGrid
from first_breaks.const import PROJECT_ROOT


def get_data_train_points():
    df = pd.read_csv(PROJECT_ROOT / "surf_forward_times_fixed.ST", skipinitialspace=True, delimiter=" ",
                     index_col=False)

    for col in ["sx", "sy", "sz", "rx", "ry", "rz", "ft"]:
        df[col] /= 1000
        print(col, df[col].min(), df[col].max())

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

    return source_point, receiver_point, travel_time_target


def train():
    x_min = 0
    x_max = 5
    z_min = 0
    z_max = 1
    nx_grid = 10
    nz_grid = 10
    max_time = 2
    max_velocity = 5
    device = "cuda"
    criterion = nn.SmoothL1Loss(0.001)

    model = TomoTransformer(in_chans=4, depth=4, embed_dim=100, num_heads=5, mlp_ratio=2, qkv_bias=True)
    model.to(device)

    source_data_points, receiver_data_points, traveltime_data_points = get_data_train_points()
    source_data_points = source_data_points.to(device)
    receiver_data_points = receiver_data_points.to(device)
    traveltime_data_points = traveltime_data_points.to(device)

    source_grid_point = get_grid_based_points(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, nx=nx_grid, nz=nz_grid)
    receiver_grid_point = get_grid_based_points(x_min=0.9 * x_max, x_max=1.1 * x_min,
                                                z_min=0.9 * z_max, z_max=1.1 * z_min,
                                                nx=nx_grid, nz=nz_grid)
    source_grid_point = source_grid_point.to(device)
    receiver_grid_point = receiver_grid_point.to(device)

    common_points_mask = (source_grid_point == receiver_grid_point).all(dim=1)
    source_grid_point = source_grid_point[~common_points_mask, ...]
    receiver_grid_point = receiver_grid_point[~common_points_mask, ...]

    print(len(source_grid_point), len(receiver_grid_point))

    num_epochs = 1000

    optimizer = Adam(model.parameters(), 5e-4)

    pbar = tqdm(range(num_epochs))

    total_loss = []

    for _ in pbar:
        # optimizer.zero_grad()

        source_data_train = source_data_points.clone().detach().requires_grad_(True)
        receiver_data_train = receiver_data_points.clone().detach().requires_grad_(True)
        # .view(-1, 4, 1)
        data_train = torch.cat([source_data_train, receiver_data_train], dim=1)

        time_data, velocity_data = model(data_train)

        time_data = max_time * sigmoid(time_data)
        # velocity_data = max_velocity * velocity_data

        loss = criterion(time_data, traveltime_data_points)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(str(loss.item()))

        total_loss.append(loss.item())

    plt.plot(total_loss)
    plt.show()
    print(min(total_loss), max(total_loss))



        # eikonal_data_pred = eikonal.get_tensors(source_data_train, receiver_data_train)
        # traveltime_data_pred, velocity_eikonal_data_pred = eikonal_data_pred["time"], eikonal_data_pred["velocity"]
        #
        # loss_data = l1_loss(traveltime_data_pred, traveltime_data_points, reduction='none')
        #
        # velocity_model_data_pred = velocity_model(receiver_data_train)
        # loss_velocity_data = l1_loss(velocity_model_data_pred, velocity_eikonal_data_pred, reduction='none')
        #
        # source_idx = torch.randperm(source_grid_point.shape[0])
        # receiver_idx = torch.randperm(receiver_grid_point.shape[0])
        #
        # source_grid_train = source_grid_point[source_idx, ...].clone().detach().requires_grad_(True)
        # receiver_grid_train = receiver_grid_point[receiver_idx, ...].clone().detach().requires_grad_(True)
        #
        # eikonal_grid_pred = eikonal.get_tensors(source_grid_train, receiver_grid_train)
        # velocity_eikonal_grid_pred = eikonal_grid_pred["velocity"]
        #
        # velocity_model_grid_pred = velocity_model(receiver_grid_train)
        # loss_velocity_grid = l1_loss(velocity_model_grid_pred, velocity_eikonal_grid_pred, reduction='none')
        #
        # total_points = sum(len(x) for x in [
        #     loss_data,
        #     loss_velocity_data,
        #     loss_velocity_grid
        # ])
        #
        # loss_data = len(loss_data) / total_points * loss_data.mean()
        # loss_velocity_data = len(loss_velocity_data) / total_points * loss_velocity_data.mean()
        # loss_velocity_grid = len(loss_velocity_grid) / total_points * loss_velocity_grid.mean()
        #
        # loss = [
        #     loss_data,
        #     loss_velocity_data,
        #     loss_velocity_grid
        # ]
        # loss = sum(loss)
        #
        # loss.backward()
        # # optimizer.step()
        #
        # optimizer_eik.step()
        # optimizer_vel.step()
        #
        # pbar.set_postfix({
        #     "loss_data": loss_data.item(),
        #     "loss_velocity_data": loss_velocity_data.item(),
        #     "loss_velocity_grid": loss_velocity_grid.item(),
        #     "eikonal_data_pred_velocity" : velocity_eikonal_data_pred.mean().item(),
        #     "velocity_model_data_pred": velocity_model_data_pred.mean().item()
        # })

    # plot_predict_on_grid(velocity_model,
    #                      x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max,
    #                      nx=nx_grid, nz=nz_grid
    #                      )


if __name__ == "__main__":
    train()
