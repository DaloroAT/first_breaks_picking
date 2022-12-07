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

    def get_slowness(self, source, receiver):
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
        self.model = torch.tensor([v], dtype=torch.float, device='cuda')
        # self.to('cuda')

    def forward(self, point):
        # if self.model.device != point.device:
        #     print(self.model.device != point.device, self.model.device, point.device)
        #     self.to(point.device)
        # print(self.model.repeat(len(point)).view(-1, 1).device)
        return self.model.repeat(len(point)).view(-1, 1)


if __name__ == '__main__':
    set_global_seed(1)
    device = 'cuda'
    dim = 2
    num_layers = 8
    hidden_dim = 20
    num_samples = 500
    lr = 1e-5
    num_epochs = 600

    eik = Eikonal(Tau(), Velocity(0.5)).to(device)
    optim = Adam(lr=lr, params=eik.parameters())

    s_train = torch.randn((num_samples, 2), device=device) / 3
    r_train = torch.randn((num_samples, 2), device=device) / 3

    s_val = torch.randn((num_samples, 2), device=device) / 3
    r_val = torch.randn((num_samples, 2), device=device) / 3

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

    t = eik(torch.tensor([[0, 0.]], device=device), torch.tensor([[0.6, 0.8]], device=device))
    print(t)





