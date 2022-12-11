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
        # self.act = nn.Sigmoid()
        # self.act = nn.ReLU()

        bias = True

        self.input_source = nn.Linear(self.dim, self.hidden_size, bias=bias)
        self.input_reciever = nn.Linear(self.dim, self.hidden_size, bias=bias)
        self.concat_input = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=bias)
        self.blocks = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=bias) for _ in range(self.num_layers)])
        self.output = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, source, receiver):
        source = self.input_source(source)
        receiver = self.input_reciever(receiver)
        features = self.act(torch.cat([source, receiver], dim=1))
        features = self.act(self.concat_input(features))

        for block in self.blocks:
            features = self.act(features) + self.act(block(features))

        features = self.act(features)

        # return torch.log1p(self.output(features))

        # features = self.act(self.output(features))
        #
        # return torch.log1p(features)

        # return torch.relu(self.output(features))
        # return self.output(self.act(features))

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

        # loss_tau_init = torch.pow(self.tau(source, source) - 1, 2)

        loss_tau_init = self.tau(source, source) - 1

        # print(torch.pow(self.step(-tau_sr) * torch.abs(tau_sr), 2).pow(2))

        loss_tau_positive = self.step(-tau_sr) * torch.abs(tau_sr)

        return loss_eikonal, loss_tau_init, loss_tau_positive

    def loss(self, source, receiver, as_floats: bool = False):
        losses = [loss.pow(2) for loss in [*self.get_losses(source, receiver), *self.get_losses(receiver, source)]]

        output = {'loss': sum(losses).sum(),
                  'loss_eikonal': (losses[0] + losses[3]).sum(),
                  'loss_tau_init': (losses[1] + losses[4]).sum(),
                  'loss_tau_positive': (losses[2] + losses[5]).sum()}

        if as_floats:
            output = {k: v.item() for k, v in output.items()}

        return output


class VelocityConst(nn.Module):
    def __init__(self, v):
        super(VelocityConst, self).__init__()
        self.model = nn.Parameter(torch.tensor([v], dtype=torch.float), requires_grad=False)

    def forward(self, point):
        return self.model.repeat(len(point)).view(-1, 1)


class VelocityVerticalLayers(nn.Module):
    def __init__(self, vel, depth, smoothing=None):
        super(VelocityVerticalLayers, self).__init__()
        assert len(vel) == len(depth) > 1
        self.vel = vel
        self.depth = depth

        vel = torch.tensor(vel, dtype=torch.float)
        depth = torch.cumsum(torch.tensor(depth, dtype=torch.float), dim=0)

        self.vel_model = nn.Parameter(vel, requires_grad=False)
        self.depth_model = nn.Parameter(depth, requires_grad=False)

        if smoothing is None:
            smoothing = (depth[1:] - depth[0:-1]).min() / 50
        self.smoothing = smoothing

    # def forward(self, point):
    #     z = point[:, -1]
    #     output = []
    #     for i, (z0_i, z1_i, v_i) in enumerate(zip([0, *self.depth_model[0:-1]], self.depth_model, self.vel_model)):
    #         if i == 0:
    #             step = torch.sigmoid((z1_i - z) / self.smoothing)
    #         elif i == len(self.vel_model) - 1:
    #             step = torch.sigmoid((z - z0_i) / self.smoothing)
    #         else:
    #             step = torch.sigmoid((z - z0_i) / self.smoothing) - torch.sigmoid((z - z1_i) / self.smoothing)
    #         output_i = v_i * step
    #         output.append(output_i)
    #     output = sum(output).view(-1, 1)
    #     return output

    def forward(self, point):
        z = point[:, -1]
        v = torch.zeros(len(z), device=point.device)

        for i, (z0_i, z1_i, v_i) in enumerate(zip([0, *self.depth_model[0:-1]], self.depth_model, self.vel_model)):
            if i == 0:
                v[z <= z1_i] = v_i
            elif i == len(self.vel_model) - 1:
                v[z >= z0_i] = v_i
            else:
                v[torch.logical_and(z >= z0_i, z < z1_i)] = v_i

        return v


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


def train(model_arg, num_samples_arg, num_epochs_arg, lr_arg, title_arg):
    optim = Adam(lr=lr_arg, params=model_arg.parameters())
    loss_train_epochs = []
    loss_val_epochs = []

    s_val = torch.rand((num_samples_arg, 2), device=device)
    r_val = torch.rand((num_samples_arg, 2), device=device)

    pbar = tqdm(range(num_epochs_arg), desc=title_arg)

    for _ in pbar:
        optim.zero_grad()

        s_train_inp = (torch.rand((num_samples_arg, 2), device=device) - 0.1).requires_grad_(True)
        r_train_inp = (torch.rand((num_samples_arg, 2), device=device) - 0.1).requires_grad_(True)

        loss = eik.loss(s_train_inp, r_train_inp)['loss']

        loss.backward()
        optim.step()

        if loss.isnan():
            raise ValueError('None!!!')

        loss_train = loss.item()

        optim.zero_grad()

        s_val_inp = s_val.clone().requires_grad_(True)
        r_val_inp = r_val.clone().requires_grad_(True)
        loss_val = eik.loss(s_val_inp, r_val_inp, as_floats=True)

        loss_train_epochs.append(loss_train)
        loss_val_epochs.append(loss_val['loss'])

        pbar.set_postfix({**loss_val,
                          'train': loss_train})

    plt.plot(loss_train_epochs, label='train')
    plt.plot(loss_val_epochs, label='val')
    plt.title(title_arg)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    set_global_seed(1)
    device = 'cuda'
    dim = 2
    num_layers = 8
    hidden_dim = 20
    num_samples = 1000
    lr = 1e-3
    num_epochs = 600

    vel_pretrain = VelocityConst(0.3).to(device)
    tau_model = Tau(hidden_size=hidden_dim, num_layers=num_layers)
    eik = Eikonal(tau_model, vel_pretrain).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_samples_arg=num_samples,
          num_epochs_arg=100,
          title_arg='pretrain')

    # vel_model = VelocityVerticalLayers([0.1, 0.2, 0.3, 0.5], [0.1, 0.1, 0.2, 0.2], smoothing=0.005).to(device)
    vel_model = VelocityVerticalLayers([0.1, 0.3], [0.1, 0.1], smoothing=0.0005).to(device)
    eik = Eikonal(tau_model, vel_model).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_samples_arg=num_samples,
          num_epochs_arg=num_epochs,
          title_arg='model')


    vel, time = visualize_maps(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), eik, 0, 0, device=device)

    plt.imshow(vel, extent=[0, 1, 1, 0])
    plt.colorbar()
    plt.show()

    plt.imshow(time, extent=[0, 1, 1, 0])
    plt.colorbar()
    plt.show()

    # vel, time = visualize_maps(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100), eik, 0.5, 0.5, device=device)
    #
    # plt.imshow(vel)
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(time)
    # plt.colorbar()
    # plt.show()







