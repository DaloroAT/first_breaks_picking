import torch
from oml.utils.misc import set_global_seed
from torch import nn
import matplotlib.pyplot as plt

from eikonal.utils import visualize_maps, train
from eikonal.velocity_models import VelocityConst, VelocityVerticalGrad, VelocityVerticalLayers


class Tau(nn.Module):
    def __init__(self, dim=2, hidden_size=20, num_layers=5):
        super(Tau, self).__init__()
        self.dim = dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        # self.act = torch.atan
        # self.act = torch.sin

        bias = True

        self.input_source = nn.Linear(self.dim, self.hidden_size, bias=bias)
        self.input_reciever = nn.Linear(self.dim, self.hidden_size, bias=bias)
        self.concat_input = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=bias)
        self.blocks = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
                                     for _ in range(self.num_layers)])
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

        for layer in [self.input_source, self.input_reciever, self.concat_input, *self.blocks, self.output]:
            nn.init.xavier_normal_(layer.weight.data, gain=1)

    def forward(self, source, receiver):
        source = self.input_source(source)
        receiver = self.input_reciever(receiver)
        features = self.act(torch.cat([source, receiver], dim=1))
        features = self.act(self.concat_input(features))

        for block in self.blocks:
            features = features + self.act(block(features))
            # features = self.act(block(features))

        features = self.act(features)

        features = self.output(features)
        # features = torch.log1p(features)
        return features


class Eikonal(nn.Module):
    def __init__(self, tau: Tau, velocity: nn.Module):
        super(Eikonal, self).__init__()
        self.tau = tau
        self.velocity = velocity

        self.step = nn.ReLU()

        self.logs_tau = []
        self.logs_t0 = []

    def forward_t0(self, source, receiver):
        dist = (receiver - source).pow(2).sum(1).sqrt().view(-1, 1)
        vel_s = self.velocity(source).view(-1, 1)
        # print('vel', vel_s.min(), vel_s.max(), 'dist', dist.min(), dist.max(), 't0', dist.max() / vel_s.min())
        return dist / vel_s

    def forward(self, source, receiver):
        tau_sr = self.tau(source, receiver)
        t0_sr = self.forward_t0(source, receiver)

        self.logs_tau.append(tau_sr.clone().detach().sum().item())
        self.logs_t0.append(t0_sr.clone().detach().sum().item())

        return tau_sr * t0_sr

    def get_velocity(self, source, receiver):
        sq_grad_eik = self.get_quare_grad_eikonal(source, receiver)
        return 1 / sq_grad_eik.abs().sqrt()

    def get_quare_grad_eikonal(self, source, receiver):
        tau_sr = self.tau(source, receiver)
        t0_sr = self.forward_t0(source, receiver)

        self.logs_tau.append(tau_sr.clone().detach().abs().mean().item())
        self.logs_t0.append(t0_sr.clone().detach().abs().mean().item())

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

        return first + second + third

    def _get_quare_grad_eikonal(self, source, receiver):
        eikonal = self(source, receiver)

        eikonal_r = torch.autograd.grad(
            eikonal,
            receiver,
            grad_outputs=torch.ones_like(eikonal),
            retain_graph=True,
            create_graph=True)[0]

        return eikonal_r.pow(2).sum(1, keepdims=True)

    def loss_eikonal(self, source, receiver):
        return self.get_quare_grad_eikonal(source, receiver) - 1 / self.velocity(receiver).pow(2)

    def loss_tau_init(self, source):
        return self.tau(source, source) - 1

    def loss_tau_positive(self, source, receiver):
        return self.step(-self.tau(source, receiver))

    def loss(self, source, receiver):
        direct_order_losses = {'eikonal': self.loss_eikonal(source, receiver).abs().mean(),
                               'tau_init': self.loss_tau_init(source).pow(2).mean(),
                               'tau_positive': self.loss_tau_positive(source, receiver).abs().mean()}

        inverse_order_losses = {'eikonal': self.loss_eikonal(receiver, source).abs().mean(),
                                'tau_init': self.loss_tau_init(receiver).pow(2).mean(),
                                'tau_positive': self.loss_tau_positive(receiver, source).abs().mean()}

        loss_to_optimize = sum([*direct_order_losses.values(), *inverse_order_losses.values()])
        # loss_to_optimize = sum([direct_order_losses['eikonal'], inverse_order_losses['eikonal']])

        losses = {"loss": loss_to_optimize,
                  **{k: (direct_order_losses[k] + inverse_order_losses[k]).sum().item()
                     for k in direct_order_losses.keys()}}

        return losses


if __name__ == '__main__':
    set_global_seed(1)
    device = 'cuda'
    dim = 2
    num_layers = 10
    hidden_dim = 20
    num_samples = 60000
    lr = 1e-2
    num_epochs = 1000

    vel_pretrain = VelocityConst(3).to(device)
    tau_model = Tau(hidden_size=hidden_dim, num_layers=num_layers, dim=dim)
    eik = Eikonal(tau_model, vel_pretrain).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_samples_arg=num_samples,
          num_epochs_arg=20,
          title_arg='pretrain',
          dim_arg=dim,
          device=device)

    # vel_model = VelocityVerticalLayers([10, 20, 30, 40], [0.1, 0.1, 0.2, 0.2]).to(device)
    # vel_model = VelocityVerticalLayers([1000, 2000, 3000, 4000], [0.1, 0.1, 0.2, 0.2]).to(device)
    # vel_model = VelocityVerticalLayers([1, 2, 3, 4], [0.2, 0.2, 0.3, 0.3], smoothing=0.001).to(device)
    # vel_model = VelocityVerticalLayers([1, 5, 15, 40], [0.2, 0.2, 0.3, 0.3], smoothing=0.001).to(device)
    # vel_model = VelocityVerticalLayers([.1, .2, .3, .4], [0.2, 0.2, 0.3, 0.3]).to(device)
    # vel_model = VelocityVerticalLayers([0.1, 0.3], [0.1, 0.1]).to(device)
    vel_model = VelocityVerticalGrad([1, 3], [0, 1]).to(device)
    eik = Eikonal(tau_model, vel_model).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_samples_arg=num_samples,
          num_epochs_arg=num_epochs,
          title_arg='model',
          dim_arg=dim,
          device=device)

    vel, time = visualize_maps(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), eik, 0, 0, device=device)
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







