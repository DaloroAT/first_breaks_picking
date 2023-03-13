from typing import Union

import torch
from oml.utils.misc import set_global_seed
from torch import nn

from eikonal.models import TauNew, Tau
from eikonal.utils import visualize_maps, train
from eikonal.velocity_models import VelocityConst, VelocityVerticalLayers


class Eikonal(nn.Module):
    def __init__(self, tau: Union[Tau, TauNew], velocity: nn.Module):
        super(Eikonal, self).__init__()
        self.tau = tau
        self.velocity = velocity

        self.step = nn.ReLU()

        self.logs_tau = []

    def forward_t0(self, source, receiver):
        dist = (receiver - source).pow(2).sum(1).sqrt().view(-1, 1)
        vel_s = self.velocity(source).view(-1, 1)
        return dist / vel_s

    def forward(self, source, receiver):
        tau_sr = self.tau(source, receiver)
        t0_sr = self.forward_t0(source, receiver)
        return tau_sr * t0_sr

    def get_velocity(self, source, receiver):
        sq_grad_eik, *_ = self.get_quare_grad_eikonal(source, receiver)
        return 1 / sq_grad_eik.abs().sqrt()

    def get_quare_grad_eikonal(self, source, receiver):
        tau_sr = self.tau(source, receiver)
        t0_sr = self.forward_t0(source, receiver)

        tau_sr_r = torch.autograd.grad(
            tau_sr,
            receiver,
            grad_outputs=torch.ones_like(tau_sr),
            retain_graph=True,
            create_graph=True)[0]

        t0_sr_r = torch.autograd.grad(
            t0_sr,
            receiver,
            grad_outputs=torch.ones_like(t0_sr),
            retain_graph=True,
            create_graph=True)[0]

        first = t0_sr.pow(2) * tau_sr_r.pow(2).sum(1, keepdims=True)
        second = tau_sr.pow(2) * t0_sr_r.pow(2).sum(1, keepdims=True)
        third = 2 * tau_sr * t0_sr * (tau_sr_r * t0_sr_r).sum(1, keepdims=True)

        return first + second + third, t0_sr, tau_sr

    def loss(self, source, receiver, weights=None):
        if weights is None:
            weights = 1
        else:
            weights = weights.view(-1, 1)
            assert len(source) == len(source)

        sq_grad_sr, t0_sr, tau_sr = self.get_quare_grad_eikonal(source, receiver)
        sq_grad_rs, t0_rs, tau_rs = self.get_quare_grad_eikonal(receiver, source)
        tau_ss = self.tau(source, source)
        tau_rr = self.tau(receiver, receiver)

        sq_v_inv_r = 1 / self.velocity(receiver).pow(2)
        sq_v_inv_s = 1 / self.velocity(source).pow(2)

        loss_tau_init_ss = (weights * (tau_ss - 1).pow(2)).mean()
        loss_tau_init_rr = (weights * (tau_rr - 1).pow(2)).mean()

        loss_tau_positive_sr = self.step(-tau_sr).mean()
        loss_tau_positive_rs = self.step(-tau_rs).mean()

        loss_eikonal_sr = (weights * (sq_grad_sr - sq_v_inv_r).abs()).mean()
        loss_eikonal_rs = (weights * (sq_grad_rs - sq_v_inv_s).abs()).mean()

        loss_eik_relative_sr = (weights * (sq_grad_sr - sq_v_inv_r).abs() * sq_v_inv_r).mean()
        loss_eikonal_relativ_rs = (weights * (sq_grad_rs - sq_v_inv_s).abs() / sq_v_inv_s).mean()

        loss_to_optimize = loss_eikonal_sr + \
                           loss_eikonal_rs + \
                           loss_tau_init_ss + \
                           loss_tau_init_rr + \
                           loss_tau_positive_sr + \
                           loss_tau_positive_rs + \
                           loss_eik_relative_sr + \
                           loss_eikonal_relativ_rs

        losses = {"loss": loss_to_optimize,
                  "eikonal": (loss_eikonal_sr + loss_eikonal_rs).item(),
                  "eik_rel": (loss_eik_relative_sr + loss_eikonal_relativ_rs).item(),
                  "tau_init": (loss_tau_init_ss + loss_tau_init_rr).item(),
                  "tau_positive": (loss_tau_positive_sr + loss_tau_positive_rs).item()}

        tau_mean = (tau_sr + tau_rs) / 2

        self.logs_tau.append(tau_mean.clone().detach().abs().mean().item())

        return losses


if __name__ == '__main__':
    set_global_seed(1)
    device = 'cuda'
    dim = 2
    num_inner_layers = 2
    num_blocks = 5
    hidden_dim = 20
    dropout = 0.0
    n_grid = 20
    lr = 3e-3
    num_epochs = 100

    vel_pretrain = VelocityConst(3).to(device)
    # tau_model = Tau(hidden_size=hidden_dim, num_layers=num_layers, dim=dim)
    tau_model = TauNew(hidden_dim=hidden_dim,
                       dim=dim,
                       num_inner_layers=num_inner_layers,
                       num_blocks=num_blocks,
                       dropout=dropout)

    # eik = Eikonal(tau_model, vel_pretrain).to(device)
    # train(model_arg=eik,
    #       lr_arg=lr,
    #       num_grid_arg=n_grid,
    #       num_epochs_arg=20,
    #       title_arg='pretrain',
    #       dim_arg=dim,
    #       device=device)

    # vel_model = VelocityVerticalLayers([10, 20, 30, 40], [0.1, 0.1, 0.2, 0.2]).to(device)
    # vel_model = VelocityVerticalLayers([1000, 2000, 3000, 4000], [0.1, 0.1, 0.2, 0.2]).to(device)
    # vel_model = VelocityVerticalLayers([1, 2, 3, 4], [0.2, 0.2, 0.3, 0.3], smoothing=0.001).to(device)
    # vel_model = VelocityVerticalLayers([1, 5, 15, 40], [0.2, 0.2, 0.3, 0.3], smoothing=0.001).to(device)
    # vel_model = VelocityVerticalLayers([.1, .2, .3, .4], [0.2, 0.2, 0.3, 0.3]).to(device)
    vel_model = VelocityVerticalLayers([0.9, 1.1], [0.5, 0.5], smoothing=0.001)
    # vel_model = VelocityVerticalLayers([0.8, 1.2, 1.6], [0.3, 0.3, 0.3], smoothing=0.001)
    # vel_model = VelocityVerticalGrad([1, 3], [0, 1]).to(device)
    vel_model.plot_vertical(0, 1)
    vel_model.to(device)
    eik = Eikonal(tau_model, vel_model).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_grid_arg=n_grid,
          num_epochs_arg=num_epochs,
          title_arg='model',
          dim_arg=dim,
          device=device)

    vel, time = visualize_maps(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), eik, 0, 0, device=device, max_vel=3)
    #
    # plt.imshow(vel, extent=[0, 1, 1, 0])
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(time, extent=[0, 1, 1, 0])
    # plt.colorbar()
    # plt.show()
    #
    # # vel, time = visualize_maps(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), eik, 0.5, 0.5, device=device)
    # #
    # # plt.imshow(vel)
    # # plt.colorbar()
    # # plt.show()
    # #
    # # plt.imshow(time)
    # # plt.colorbar()
    # # plt.show()







