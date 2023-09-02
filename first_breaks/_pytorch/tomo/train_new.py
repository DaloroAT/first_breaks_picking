import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import mse_loss, l1_loss
from torch.optim import Adam
from tqdm import tqdm

from first_breaks._pytorch.tomo.models import Tau, Eikonal, Model
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


def _train_eikonal(eikonal_model, velocity_model, optimizer, num_steps, source_data_point, receiver_data_point, traveltime_data_point, source_grid_point, receiver_grid_point):
    # pbar = tqdm(range(num_steps), desc="Eikonal stage", position=1)

    criterion_data = nn.MSELoss(reduction='none')
    criterion_grid = nn.SmoothL1Loss(beta=0.2, reduction='none')

    for _ in range(num_steps):
        optimizer.zero_grad()
        source_data_train = source_data_point.clone().detach().requires_grad_(True)
        receiver_data_train = receiver_data_point.clone().detach().requires_grad_(True)

        source_idx = torch.randperm(source_grid_point.shape[0])
        receiver_idx = torch.randperm(receiver_grid_point.shape[0])
        source_grid_train = source_grid_point[source_idx, ...].clone().detach().requires_grad_(True)
        receiver_grid_train = receiver_grid_point[receiver_idx, ...].clone().detach().requires_grad_(True)

        time_data_pred = eikonal_model(source_data_train, receiver_data_train)
        loss_data_raw = criterion_data(time_data_pred, traveltime_data_point)

        velocity_model_grid_pred = velocity_model(receiver_grid_train.clone().detach().requires_grad_(True))
        velocity_model.zero_grad()

        velocity_eikonal_grid_pred = eikonal_model.get_tensors(source_grid_train, receiver_grid_train)["velocity"]

        loss_grid_raw = criterion_grid(velocity_model_grid_pred, velocity_eikonal_grid_pred)

        total_samples = sum(len(x) for x in [
            loss_data_raw,
            loss_grid_raw
        ])

        loss_data = len(loss_data_raw) / total_samples * loss_data_raw.mean()
        loss_grid = len(loss_grid_raw) / total_samples * loss_grid_raw.mean()

        loss_total = 3 * loss_data + loss_grid

        loss_total.backward()
        optimizer.step()

        # pbar.set_postfix({"loss_total": loss_total.item(),
        #                   "loss_data_raw": loss_data_raw.mean().item(),
        #                   "loss_data_sc": loss_data.mean().item(),
        #                   "loss_grid_vel_raw": loss_grid_raw.mean().item(),
        #                   "loss_grid_vel_sc": loss_grid.mean().item(),
        #                   "velocity_model_grid_pred": velocity_model_grid_pred.mean().item(),
        #                   "velocity_eikonal_grid_pred": velocity_eikonal_grid_pred.mean().item()
        #                   })

        yield {"loss_data": loss_data_raw.mean().item(),
               "velocity_difference": velocity_model_grid_pred.mean().item() - velocity_eikonal_grid_pred.mean().item(),
               "velocity_model_grid_pred": velocity_model_grid_pred.mean().item(),
               "velocity_eikonal_grid_pred": velocity_eikonal_grid_pred.mean().item()
               }


def _train_velocity(velocity_model, eikonal_model, optimizer, num_steps, source_grid_point, receiver_grid_point):
    # pbar = tqdm(range(num_steps), desc="Velocity stage", position=2)

    criterion_grid = nn.SmoothL1Loss(beta=0.2, reduction='mean')

    for _ in range(num_steps):
        optimizer.zero_grad()

        source_idx = torch.randperm(source_grid_point.shape[0])
        receiver_idx = torch.randperm(receiver_grid_point.shape[0])
        source_grid_train = source_grid_point[source_idx, ...].clone().detach().requires_grad_(True)
        receiver_grid_train = receiver_grid_point[receiver_idx, ...].clone().detach().requires_grad_(True)

        velocity_model_grid_pred = velocity_model(receiver_grid_train)

        velocity_eikonal_grid_pred = eikonal_model.get_tensors(source_grid_train, receiver_grid_train)["velocity"]

        loss_grid = criterion_grid(velocity_eikonal_grid_pred, velocity_model_grid_pred)

        loss_grid.backward()
        optimizer.step()

        # pbar.set_postfix({"loss_grid_vel": loss_grid.item(),
        #                   "velocity_model_grid_pred": velocity_model_grid_pred.mean().item(),
        #                   "velocity_eikonal_grid_pred": velocity_eikonal_grid_pred.mean().item()
        #                   })

        yield {"velocity_difference": velocity_model_grid_pred.mean().item() - velocity_eikonal_grid_pred.mean().item(),
               "velocity_model_grid_pred": velocity_model_grid_pred.mean().item(),
               "velocity_eikonal_grid_pred": velocity_eikonal_grid_pred.mean().item()
               }


def train():
    x_min = 0
    x_max = 5
    z_min = 0
    z_max = 1

    nz_grid = 10
    nx_grid = 5 * nz_grid

    tau_model = Tau(input_dim=4, hidden_dim=20, num_layers=10, max_value=2)
    velocity_model = Model(input_dim=2, hidden_dim=10, num_layers=10, max_value=5)
    # velocity_model = VelocityGrid(vel_grid=4 * torch.ones((nz_grid, nx_grid)),
    #                               x_grid=torch.linspace(x_min, x_max, nx_grid),
    #                               z_grid=torch.linspace(z_min, z_max, nz_grid),
    #                               use_weighted_forward=False,
    #                               learnable_cell_size=False)

    eikonal = Eikonal(tau_model=tau_model, background_velocity=None)

    source_data_points, receiver_data_points, traveltime_data_points = get_data_train_points()

    source_grid_point = get_grid_based_points(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, nx=nx_grid, nz=nz_grid)
    receiver_grid_point = get_grid_based_points(x_min=0.9 * x_max, x_max=1.1 * x_min,
                                                z_min=0.9 * z_max, z_max=1.1 * z_min,
                                                nx=nx_grid, nz=nz_grid)
    common_points_mask = (source_grid_point == receiver_grid_point).all(dim=1)
    source_grid_point = source_grid_point[~common_points_mask, ...]
    receiver_grid_point = receiver_grid_point[~common_points_mask, ...]

    print(len(source_grid_point), len(receiver_grid_point))

    num_epochs = 100

    optimizer_eik = Adam(eikonal.parameters(), 1e-2)
    optimizer_vel = Adam(velocity_model.parameters(), 1e-2)

    pbar_epoch = tqdm(range(num_epochs), desc='Epoch')

    n_steps_eikonal = 5
    n_steps_velocity = 5

    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"Epoch {epoch}")
        logs_eikonal = {}
        logs_velocity = {}

        trainloop_eikonal = _train_eikonal(eikonal_model=eikonal,
                                           velocity_model=velocity_model,
                                           optimizer=optimizer_eik,
                                           num_steps=n_steps_eikonal,
                                           source_data_point=source_data_points,
                                           receiver_data_point=receiver_data_points,
                                           traveltime_data_point=traveltime_data_points,
                                           source_grid_point=source_grid_point,
                                           receiver_grid_point=receiver_grid_point)
        for logs_eikonal in trainloop_eikonal:
            pbar_epoch.set_postfix({**logs_eikonal, **logs_velocity})

        trainloop_velocity = _train_velocity(eikonal_model=eikonal,
                                             velocity_model=velocity_model,
                                             optimizer=optimizer_vel,
                                             num_steps=n_steps_velocity,
                                             source_grid_point=source_grid_point,
                                             receiver_grid_point=receiver_grid_point)
        for logs_velocity in trainloop_velocity:
            pbar_epoch.set_postfix({**logs_eikonal, **logs_velocity})

    velocity_model.eval()
    plot_predict_on_grid(velocity_model,
                         x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max,
                         nx=nx_grid, nz=nz_grid,
                         title="Velocity")
    eikonal.eval()
    plot_predict_on_grid(lambda x: eikonal.forward(torch.zeros(nz_grid * nx_grid, 2), x),
                         x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max,
                         nx=nx_grid, nz=nz_grid,
                         title="Time")




# def train():
#     x_min = 0
#     x_max = 5
#     z_min = 0
#     z_max = 1
#     nx_grid = 50
#     nz_grid = 50
#
#     tau_model = Tau(input_dim=4, hidden_dim=20, num_layers=10, max_value=3)
#     velocity_model = Model(input_dim=2, hidden_dim=10, num_layers=10, max_value=5)
#     # velocity_model = VelocityGrid(vel_grid=4 * torch.ones((nz_grid, nx_grid)),
#     #                               x_grid=torch.linspace(x_min, x_max, nx_grid),
#     #                               z_grid=torch.linspace(z_min, z_max, nz_grid),
#     #                               use_weighted_forward=False,
#     #                               learnable_cell_size=False)
#
#     eikonal = Eikonal(tau_model=tau_model, background_velocity=None)
#
#     source_data_points, receiver_data_points, traveltime_data_points = get_data_train_points()
#
#     source_grid_point = get_grid_based_points(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, nx=nx_grid, nz=nz_grid)
#     receiver_grid_point = get_grid_based_points(x_min=0.9 * x_max, x_max=1.1 * x_min,
#                                                 z_min=0.9 * z_max, z_max=1.1 * z_min,
#                                                 nx=nx_grid, nz=nz_grid)
#     common_points_mask = (source_grid_point == receiver_grid_point).all(dim=1)
#     source_grid_point = source_grid_point[~common_points_mask, ...]
#     receiver_grid_point = receiver_grid_point[~common_points_mask, ...]
#
#     print(len(source_grid_point), len(receiver_grid_point))
#
#     num_epochs = 100
#
#     # lr = 1e-2
#     # optimizer = Adam(list(eikonal.parameters()) + list(velocity_model.parameters()), lr)
#     # optimizer = Adam(eikonal.parameters(), lr)
#
#     optimizer_eik = Adam(eikonal.parameters(), 1e-2)
#     optimizer_vel = Adam(velocity_model.parameters(), 1e-2)
#
#     pbar = tqdm(range(num_epochs))
#
#     for _ in pbar:
#         optimizer_eik.zero_grad()
#         optimizer_vel.zero_grad()
#         # optimizer.zero_grad()
#
#         source_data_train = source_data_points.clone().detach().requires_grad_(False)
#         receiver_data_train = receiver_data_points.clone().detach().requires_grad_(True)
#
#         eikonal_data_pred = eikonal.get_tensors(source_data_train, receiver_data_train)
#         traveltime_data_pred = eikonal_data_pred["time"]
#
#         loss_data = l1_loss(traveltime_data_pred, traveltime_data_points, reduction='none')
#
#         source_idx = torch.randperm(source_grid_point.shape[0])
#         receiver_idx = torch.randperm(receiver_grid_point.shape[0])
#
#         source_grid_train = source_grid_point[source_idx, ...].clone().detach().requires_grad_(True)
#         receiver_grid_train = receiver_grid_point[receiver_idx, ...].clone().detach().requires_grad_(True)
#
#         eikonal_grid_pred = eikonal.get_tensors(source_grid_train, receiver_grid_train)
#         square_slowness_eikonal_grid_pred, velocity_eikonal_grid_pred = eikonal_grid_pred["square_slowness"], eikonal_grid_pred["velocity"]
#
#         velocity_model_grid_pred = velocity_model(receiver_grid_train)
#         square_slowness_model_grid_pred = 1 / velocity_model_grid_pred ** 2
#         loss_square_slowness_grid = l1_loss(square_slowness_model_grid_pred, square_slowness_eikonal_grid_pred, reduction='none')
#
#         total_points = sum(len(x) for x in [
#             loss_data,
#             # loss_square_slowness_grid,
#         ])
#
#         loss_data = len(loss_data) / total_points * loss_data.mean()
#         # loss_square_slowness_grid = len(loss_square_slowness_grid) / total_points * loss_square_slowness_grid.mean()
#
#         loss = [
#             loss_data,
#             # loss_square_slowness_grid,
#         ]
#         loss = sum(loss)
#
#         loss.backward()
#         # optimizer.step()
#
#         optimizer_eik.step()
#         optimizer_vel.step()
#
#         pbar.set_postfix({
#             "loss_data": loss_data.item(),
#             # "loss_square_slowness_grid": loss_square_slowness_grid.item(),
#             "velocity_eikonal_grid_pred" : velocity_eikonal_grid_pred.mean().item(),
#             "velocity_model_grid_pred": velocity_model_grid_pred.mean().item()
#         })
#
#     plot_predict_on_grid(velocity_model,
#                          x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max,
#                          nx=nx_grid, nz=nz_grid
#                          )


#
# def train_1():
#     x_min = 0
#     x_max = 5
#     z_min = 0
#     z_max = 1
#     nx_grid = 10
#     nz_grid = 10
#
#     tau_model = Tau(input_dim=4, hidden_dim=20, num_layers=10, max_value=2)
#     # velocity_model = Model(input_dim=2, hidden_dim=10, num_layers=10, max_value=5)
#     velocity_model = VelocityGrid(vel_grid=4 * torch.ones((nz_grid, nx_grid)),
#                                   x_grid=torch.linspace(x_min, x_max, nx_grid),
#                                   z_grid=torch.linspace(z_min, z_max, nz_grid),
#                                   use_weighted_forward=False,
#                                   learnable_cell_size=False)
#
#     eikonal = Eikonal(tau_model=tau_model, background_velocity=None)
#
#     source_data_points, receiver_data_points, traveltime_data_points = get_data_train_points()
#
#     source_grid_point = get_grid_based_points(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, nx=nx_grid, nz=nz_grid)
#     receiver_grid_point = get_grid_based_points(x_min=0.9 * x_max, x_max=1.1 * x_min,
#                                                 z_min=0.9 * z_max, z_max=1.1 * z_min,
#                                                 nx=nx_grid, nz=nz_grid)
#     common_points_mask = (source_grid_point == receiver_grid_point).all(dim=1)
#     source_grid_point = source_grid_point[~common_points_mask, ...]
#     receiver_grid_point = receiver_grid_point[~common_points_mask, ...]
#
#     print(len(source_grid_point), len(receiver_grid_point))
#
#     num_epochs = 100
#
#     # lr = 1e-2
#     # optimizer = Adam(list(eikonal.parameters()) + list(velocity_model.parameters()), lr)
#     # optimizer = Adam(eikonal.parameters(), lr)
#
#     optimizer_eik = Adam(eikonal.parameters(), 1e-2)
#     optimizer_vel = Adam(velocity_model.parameters(), 1e-1)
#
#     pbar = tqdm(range(num_epochs))
#
#     for _ in pbar:
#         optimizer_eik.zero_grad()
#         optimizer_vel.zero_grad()
#         # optimizer.zero_grad()
#
#         source_data_train = source_data_points.clone().detach().requires_grad_(False)
#         receiver_data_train = receiver_data_points.clone().detach().requires_grad_(True)
#
#         eikonal_data_pred = eikonal.get_tensors(source_data_train, receiver_data_train)
#         traveltime_data_pred, velocity_eikonal_data_pred = eikonal_data_pred["time"], eikonal_data_pred["velocity"]
#
#         loss_data = l1_loss(traveltime_data_pred, traveltime_data_points, reduction='none')
#
#         velocity_model_data_pred = velocity_model(receiver_data_train)
#         loss_velocity_data = l1_loss(velocity_model_data_pred, velocity_eikonal_data_pred, reduction='none')
#
#         source_idx = torch.randperm(source_grid_point.shape[0])
#         receiver_idx = torch.randperm(receiver_grid_point.shape[0])
#
#         source_grid_train = source_grid_point[source_idx, ...].clone().detach().requires_grad_(True)
#         receiver_grid_train = receiver_grid_point[receiver_idx, ...].clone().detach().requires_grad_(True)
#
#         eikonal_grid_pred = eikonal.get_tensors(source_grid_train, receiver_grid_train)
#         velocity_eikonal_grid_pred = eikonal_grid_pred["velocity"]
#
#         velocity_model_grid_pred = velocity_model(receiver_grid_train)
#         loss_velocity_grid = l1_loss(velocity_model_grid_pred, velocity_eikonal_grid_pred, reduction='none')
#
#         total_points = sum(len(x) for x in [
#             loss_data,
#             loss_velocity_data,
#             loss_velocity_grid
#         ])
#
#         loss_data = len(loss_data) / total_points * loss_data.mean()
#         loss_velocity_data = len(loss_velocity_data) / total_points * loss_velocity_data.mean()
#         loss_velocity_grid = len(loss_velocity_grid) / total_points * loss_velocity_grid.mean()
#
#         loss = [
#             loss_data,
#             loss_velocity_data,
#             loss_velocity_grid
#         ]
#         loss = sum(loss)
#
#         loss.backward()
#         # optimizer.step()
#
#         optimizer_eik.step()
#         optimizer_vel.step()
#
#         pbar.set_postfix({
#             "loss_data": loss_data.item(),
#             "loss_velocity_data": loss_velocity_data.item(),
#             "loss_velocity_grid": loss_velocity_grid.item(),
#             "eikonal_data_pred_velocity" : velocity_eikonal_data_pred.mean().item(),
#             "velocity_model_data_pred": velocity_model_data_pred.mean().item()
#         })
#
#     plot_predict_on_grid(velocity_model,
#                          x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max,
#                          nx=nx_grid, nz=nz_grid
#                          )


if __name__ == "__main__":
    train()
