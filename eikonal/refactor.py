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
        features = torch.log1p(features)
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
                                'tau_positive': self.loss_tau_positive(receiver, source).abs()
                                }

        # loss_to_optimize = sum([*direct_order_losses.values(), *inverse_order_losses.values()]).sum()
        loss_to_optimize = sum([direct_order_losses['eikonal'], inverse_order_losses['eikonal']])

        losses = {"loss": loss_to_optimize,
                  **{k: (direct_order_losses[k] + inverse_order_losses[k]).sum().item()
                     for k in direct_order_losses.keys()}}

        return losses


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

    def forward(self, point):
        z = point[:, -1]
        output = []
        for i, (z0_i, z1_i, v_i) in enumerate(zip([0, *self.depth_model[0:-1]], self.depth_model, self.vel_model)):
            if i == 0:
                step = torch.sigmoid((z1_i - z) / self.smoothing)
            elif i == len(self.vel_model) - 1:
                step = torch.sigmoid((z - z0_i) / self.smoothing)
            else:
                step = torch.sigmoid((z - z0_i) / self.smoothing) - torch.sigmoid((z - z1_i) / self.smoothing)
            output_i = v_i * step
            output.append(output_i)
        output = sum(output).view(-1, 1)
        return output


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


def train(model_arg, num_samples_arg, num_epochs_arg, lr_arg, title_arg, dim_arg):
    optim = Adam(lr=lr_arg, params=model_arg.parameters())
    loss_train_epochs = []
    loss_val_epochs = []

    s_val = torch.rand((num_samples_arg, dim_arg), device=device)
    r_val = torch.rand((num_samples_arg, dim_arg), device=device)

    pbar = tqdm(range(num_epochs_arg), desc=title_arg)

    for _ in pbar:

        s_train_inp = (torch.rand((num_samples_arg, dim_arg), device=device)).requires_grad_(True)
        r_train_inp = (torch.rand((num_samples_arg, dim_arg), device=device)).requires_grad_(True)
        loss = eik.loss(s_train_inp, r_train_inp)['loss']

        loss.backward()
        optim.step()

        if loss.isnan():
            raise ValueError('None!!!')

        loss_train = loss.item()

        optim.zero_grad()

        s_val_inp = s_val.clone().requires_grad_(True)
        r_val_inp = r_val.clone().requires_grad_(True)
        loss_val = eik.loss(s_val_inp, r_val_inp)

        loss_train_epochs.append(loss_train)
        loss_val['loss'] = loss_val['loss'].item()
        loss_val_epochs.append(loss_val['loss'])

        pbar.set_postfix({**loss_val,
                          'train': loss_train})

    plt.plot(loss_train_epochs, label='train')
    plt.plot(loss_val_epochs, label='val')
    plt.title(title_arg)
    plt.legend()
    plt.show()

    plt.plot(eik.logs_tau)
    plt.title('tau')
    plt.show()
    plt.plot(eik.logs_t0)
    plt.title('t0')
    plt.show()


if __name__ == '__main__':
    set_global_seed(1)
    device = 'cuda'
    dim = 2
    num_layers = 5
    hidden_dim = 20
    num_samples = 40000
    lr = 1e-3
    num_epochs = 1000

    vel_pretrain = VelocityConst(3).to(device)
    tau_model = Tau(hidden_size=hidden_dim, num_layers=num_layers, dim=dim)
    eik = Eikonal(tau_model, vel_pretrain).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_samples_arg=num_samples,
          num_epochs_arg=20,
          title_arg='pretrain',
          dim_arg=dim)

    vel_model = VelocityVerticalLayers([10, 20, 30, 40], [0.1, 0.1, 0.2, 0.2]).to(device)
    # vel_model = VelocityVerticalLayers([1000, 2000, 3000, 4000], [0.1, 0.1, 0.2, 0.2]).to(device)
    # vel_model = VelocityVerticalLayers([1, 2, 3, 4], [0.2, 0.2, 0.3, 0.3]).to(device)
    # vel_model = VelocityVerticalLayers([0.1, 0.3], [0.1, 0.1]).to(device)
    eik = Eikonal(tau_model, vel_model).to(device)

    train(model_arg=eik,
          lr_arg=lr,
          num_samples_arg=num_samples,
          num_epochs_arg=num_epochs,
          title_arg='model',
          dim_arg=dim)

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







