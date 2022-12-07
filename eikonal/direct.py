import torch
from oml.utils.misc import set_global_seed
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm


class Tau(nn.Module):
    def __init__(self, dim=2, hidden_size=20, num_layers=5):
        super(Tau, self).__init__()
        self.dim = dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.act = nn.Tanh()

        self.input_source = nn.Linear(self.dim, self.hidden_size)
        self.input_reciever = nn.Linear(self.dim, self.hidden_size)
        self.concat_input = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.blocks = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])
        self.output = nn.Linear(self.hidden_size, 1)

    def forward(self, source, receiver):
        source = self.input_source(source)
        receiver = self.input_reciever(receiver)
        features = self.act(torch.cat([source, receiver], dim=1))
        features = self.act(self.concat_input(features))

        for block in self.blocks:
            features = features + self.act(block(features))

        features = self.act(features)

        return self.output(features)


class Eikonal(nn.Module):
    def __init__(self, tau: Tau, velocity: nn.Module):
        super(Eikonal, self).__init__()
        self.tau = tau
        self.velocity = velocity

        self.step = nn.ReLU()

    def forward_t0(self, source, receiver):
        dist = (receiver - source).pow(2).sum(1).sqrt().view(-1, 1)
        vel_s = self.velocity(source).view(-1, 1)
        return dist / vel_s

    def forward(self, source, receiver):
        return self.tau(source, receiver) * self.forward_t0(source, receiver)

    def get_velocity(self, source, receiver):
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

        first = t0_sr.pow(2) * tau_sr_r.pow(2).sum(1).view(-1, 1)
        second = tau_sr.pow(2) * t0_sr_r.pow(2).sum(1).view(-1, 1)
        third = 2 * (t0_sr_r * tau_sr_r).sum(1).view(-1, 1) * tau_sr * t0_sr

        return 1 / (first + second + third).abs().sqrt()

    def get_losses(self, source, receiver):
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

        first = t0_sr.pow(2) * tau_sr_r.pow(2).sum(1).view(-1, 1)
        second = tau_sr.pow(2) * t0_sr_r.pow(2).sum(1).view(-1, 1)
        third = 2 * (t0_sr_r * tau_sr_r).sum(1).view(-1, 1) * tau_sr * t0_sr
        loss_eikonal = first + second + third - 1 / self.velocity(receiver).pow(2)

        loss_tau_init = self.tau(source, source) - 1

        loss_tau_positive = self.step(-tau_sr) * torch.abs(tau_sr)

        return loss_eikonal, loss_tau_init, loss_tau_positive

    def loss(self, source, receiver):
        # loss_eik = sum([*self.get_losses(source, receiver), *self.get_losses(receiver, source)])
        losses = sum([loss.pow(2) for loss in [*self.get_losses(source, receiver), *self.get_losses(receiver, source)]])
        loss_eik = losses.sum()
        return loss_eik


class Velocity(nn.Module):
    def __init__(self, v):
        super(Velocity, self).__init__()
        self.model = nn.Parameter(torch.tensor([v], dtype=torch.float), requires_grad=False)

    def forward(self, point):
        return self.model.repeat(len(point)).view(-1, 1)


def visualize_maps(x_vec, z_vec, model, x0, z0, device):
    x_vec = x_vec.clone().float().to(device)
    z_vec = z_vec.clone().float().to(device)
    xx, zz = torch.meshgrid([x_vec, z_vec])
    xx = xx.flatten().view(-1, 1)
    zz = zz.flatten().view(-1, 1)

    source = torch.tensor([[x0, z0]], dtype=torch.float).repeat(len(xx), 1).to(device)
    receiver = torch.cat([xx, zz], dim=1)

    vel = model.get_velocity(source.clone().requires_grad_(True),
                             receiver.clone().requires_grad_(True)).detach().squeeze().cpu().numpy()

    vel_map = vel.reshape((len(x_vec), len(z_vec)), order='F')

    time = model(source.clone().requires_grad_(True),
                 receiver.clone().requires_grad_(True)).detach().squeeze().cpu().numpy()

    time = time.reshape((len(x_vec), len(z_vec)), order='F')
    return vel_map, time


if __name__ == '__main__':
    set_global_seed(1)
    device = 'cuda'
    dim = 2
    num_layers = 8
    hidden_dim = 20
    num_samples = 500
    lr = 1e-2
    num_epochs = 40

    eik = Eikonal(Tau(), Velocity(0.2).to(device)).to(device)
    optim = Adam(lr=lr, params=eik.parameters())

    s_train = torch.randn((num_samples, 2), device=device)
    r_train = torch.randn((num_samples, 2), device=device)

    s_val = torch.randn((num_samples, 2), device=device)
    r_val = torch.randn((num_samples, 2), device=device)

    loss_train_epochs = []
    loss_val_epochs = []

    pbar = tqdm(range(num_epochs))

    for _ in pbar:
        optim.zero_grad()

        s_train_inp = s_train.clone().requires_grad_(True)
        r_train_inp = r_train.clone().requires_grad_(True)
        loss = eik.loss(s_train_inp, r_train_inp)

        loss.backward()
        optim.step()

        loss_train = loss.item()

        s_val_inp = s_val.clone().requires_grad_(True)
        r_val_inp = r_val.clone().requires_grad_(True)
        loss_val = eik.loss(s_val_inp, r_val_inp).item()

        loss_train_epochs.append(loss_train)
        loss_val_epochs.append(loss_val)

    plt.plot(loss_train_epochs, label='train')
    plt.plot(loss_val_epochs, label='val')
    plt.legend()
    plt.show()

    vel, time = visualize_maps(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), eik, 0, 0, device=device)

    plt.imshow(vel)
    plt.colorbar()
    plt.show()

    plt.imshow(time)
    plt.colorbar()
    plt.show()

    vel, time = visualize_maps(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), eik, 0.5, 0.5, device=device)

    plt.imshow(vel)
    plt.colorbar()
    plt.show()

    plt.imshow(time)
    plt.colorbar()
    plt.show()







