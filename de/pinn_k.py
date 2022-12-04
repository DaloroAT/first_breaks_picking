from math import factorial
from operator import itemgetter

import cv2
import torch
from matplotlib.animation import FuncAnimation
from oml.utils.misc import set_global_seed
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from tqdm import tqdm


plt.rcParams.update({'font.size': 18})


# class PolyNN(nn.Module):
#     def __init__(self, polynomial_degree, is_pinn):
#         super().__init__()
#         self.polynomial_degree = polynomial_degree
#         self.coeffs = nn.Linear(polynomial_degree, 1, bias=False)
#         self.init_with_const(0.1)
#
#         self.is_pinn = is_pinn
#
#     def init_with_const(self, const=0.0):
#         self.coeffs.weight.data = (const * torch.ones(self.polynomial_degree)).requires_grad_(True)[None, :]
#
#     def forward(self, t):
#         features = []
#         for degree in range(0, self.polynomial_degree):
#             features.append(t ** degree)
#         features = torch.cat(features, dim=1)
#
#         return self.coeffs(features)


# class PolyNN(nn.Module):
#     def __init__(self, polynomial_degree, is_pinn):
#         super().__init__()
#         self.polynomial_degree = polynomial_degree
#         self.coeffs = nn.Linear(polynomial_degree - 1, 1, bias=True)
#         self.init_with_const(0.1)
#
#         self.is_pinn = is_pinn
#
#     def init_with_const(self, const=0.0):
#         self.coeffs.bias.data = torch.zeros_like(self.coeffs.bias.data)
#         self.coeffs.weight.data = (const * torch.ones(self.polynomial_degree - 1)).requires_grad_(True)[None, :]
#
#     def forward(self, x):
#         features = []
#         for degree in range(1, self.polynomial_degree):
#             features.append(x ** degree)
#         features = torch.cat(features, dim=1)
#
#         return self.coeffs(features)
#
#     def loss(self, x, gt, x_ode=None):
#         if self.is_pinn:
#             return self._loss_pinn(x, gt, x_ode)
#         else:
#             return self._loss_vanila(x, gt, x_ode)


# class PolyNN(nn.Module):
#     def __init__(self, polynomial_degree):
#         super().__init__()
#         self.polynomial_degree = polynomial_degree
#         self.coeffs = nn.Linear(polynomial_degree, 1, bias=False)
#         self.init_with_const(0.1)
#
#     def forward(self, t):
#         features = []
#         for degree in range(0, self.polynomial_degree):
#             features.append(t ** degree)
#         features = torch.cat(features, dim=1)
#
#         return self.coeffs(features)
#
#     def loss(self, t, gt):
#         x = self(t)
#         loss = torch.sum(torch.pow(gt - x, 2))
#         return loss


class PolyNN(nn.Module):
    def __init__(self, polynomial_degree):
        super().__init__()
        self.polynomial_degree = polynomial_degree
        self.coeffs = nn.Linear(polynomial_degree, 1, bias=False)
        self.w2 = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.amp = 1
        self.init_with_const(0.1)

        self.is_pinn = is_pinn
        self.w2 = nn.Parameter(torch.tensor([1.7], requires_grad=True))

    def init_with_const(self, const=0.0):
        self.coeffs.weight.data = (const * torch.ones(self.polynomial_degree)).requires_grad_(True)[None, :]

    def forward(self, t):
        features = []
        for degree in range(0, self.polynomial_degree):
            features.append(t ** degree)
        features = torch.cat(features, dim=1)

        return self.coeffs(features)

    def loss(self, t, gt, t_ode=None):
        if self.is_pinn:
            return self._loss_pinn(t, gt, t_ode)
        else:
            return self._loss_l2(t, gt)

    def _loss_l2(self, t, gt):
        x = self(t)
        loss = torch.sum(torch.pow(gt - x, 2))
        return loss

    def _loss_motion(self, t_pinn):
        x = self(t_pinn)
        x_t = torch.autograd.grad(
            x,
            t_pinn,
            grad_outputs=torch.ones_like(x),
            retain_graph=True, create_graph=True)[0]
        x_tt = torch.autograd.grad(
            x_t,
            t_pinn,
            grad_outputs=torch.ones_like(x_t),
            retain_graph=True, create_graph=True)[0]

        motion = x_tt + self.w2 * x
        return torch.sum(torch.pow(motion, 2))

    def _loss_energy(self, t_pinn):
        x = self(t_pinn)
        x_t = torch.autograd.grad(
            x,
            t_pinn,
            grad_outputs=torch.ones_like(x),
            retain_graph=True, create_graph=True)[0]
        energy = x_t ** 2 + self.w2 * x ** 2 - self.w2 * self.amp ** 2
        return torch.sum(torch.pow(energy, 2))

    def _loss_pinn(self, t, gt, t_pinn=None):
        if t_pinn is None:
            t_pinn = t

        a = len(t) / len(t_pinn) / 2

        losses = (
            self._loss_l2(t, gt),
            a * self._loss_newton2(t_pinn),
            a * self._loss_energy(t_pinn),
        )

        return sum(losses)

    def loss_total(self, t, gt, t_pinn):
        a = len(t) / len(t_pinn) / 2

        losses = (
            self._loss_l2(t, gt),
            a * self._loss_newton2(t_pinn),
            a * self._loss_energy(t_pinn),
        )

        return sum(losses)


def make_animation(data_train_arg, gt_train_arg, data_val_arg, gt_val_arg, plot_vanila_arg, plot_pinn_arg, w2_arg,
                   preds_pinn_arg, preds_vanila_arg, fname_arg, fps_arg=10, title_arg='', interval_arg=1):
    fig = plt.figure(figsize=(16, 9))
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlim([min(data_val_arg), max(data_val_arg)])
    axes.set_ylim([min(gt_val_arg) - 0.5, max(gt_val_arg) + 0.5])

    plt.xlabel("t")
    plt.ylabel("x")

    t = plt.title(title_arg, fontdict={'size': 14})

    plt.plot(data_val_arg, gt_val_arg, color=(0, 0, 0, 0.3), linewidth=3, label='sin(t)')
    plt.scatter(data_train_arg, gt_train_arg, color=(3 / 256, 125 / 256, 80 / 256), linewidth=4, label='t_train')

    if plot_pinn_arg:
        pinn_line = plt.plot(data_val_arg, [None] * len(data_val_arg), color='r', label='PINN', linewidth=3)[0]
    if plot_vanila_arg:
        vanila_line = plt.plot(data_val_arg, [None] * len(data_val_arg), color='b', label='Vanila', linewidth=3)[0]

    plt.legend(loc='upper right', prop={'size': 14})

    fig.tight_layout()

    frames = len(preds_pinn_arg) // interval_arg

    pbar = tqdm('Saving animation', total=frames)

    def animation(step):
        step = step * interval_arg
        step_title = (title_arg + f" , epoch={step}, w2={round(w2_arg[step].item(), 3)}").strip(' , ')
        t.set_text(step_title)
        if plot_pinn_arg:
            pinn_line.set_ydata(preds_pinn_arg[step, :])
        if plot_vanila_arg:
            vanila_line.set_ydata(preds_vanila_arg[step, :])
        assert step < len(preds_pinn_arg)
        pbar.update(1)

    animation = FuncAnimation(fig, animation, frames=frames)
    animation.save(fname_arg, fps=fps_arg)
    pbar.close()


def func(dataset):
    # return torch.sin(dataset * torch.pi)
    return torch.sin(dataset)


def train(model_arg, data_train_arg, gt_train_arg, data_val_arg,
          lr_arg, epochs_arg, noise_arg=0):
    optimizer = Adam(lr=lr_arg, params=model_arg.parameters())

    pbar = tqdm(range(epochs_arg))

    extra_noise = noise_arg * torch.randn(data_train_arg.shape, device=device)

    full_preds_val = torch.zeros((epochs_arg, len(data_val_arg)), device=device)
    data_ode_raw = torch.linspace(min(data_val_arg), max(data_val_arg), 200, device=device)
    w2_model = torch.zeros(epochs_arg)

    for epoch in pbar:
        optimizer.zero_grad()
        data = data_train_arg.clone().detach().requires_grad_(True)[:, None]
        gt = (gt_train_arg + extra_noise).clone().detach().requires_grad_(True)[:, None]
        data_ode = data_ode_raw.clone().detach().requires_grad_(True)[:, None]
        # data_ode = None

        loss = model_arg.loss(data, gt, data_ode)

        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(loss.item())

        # if loss.item() > 100 and epoch > 1000:
        #     assert False, epoch

        with torch.no_grad():
            data_val_epoch = data_val_arg.clone().detach().requires_grad_(True)[:, None]

            preds_val = model_arg(data_val_epoch)
            full_preds_val[epoch, :] = preds_val.squeeze()

        w2_model[epoch] = model_arg.w2.data.item()

    return full_preds_val.cpu(), w2_model.cpu()


if __name__ == "__main__":
    set_global_seed(1)
    device = 'cpu'

    lr = 4e-2
    num_points = 40
    drop_rate = 0.85
    epochs = 21800
    noise = 0
    fps = 20
    interval = 100
    plot_pinn = True
    plot_vanila = False

    diap = [-torch.pi, torch.pi]

    data_val = torch.linspace(*diap, 1000, device=device)
    gt_val = func(data_val)
    data_full = torch.linspace(*diap, num_points, device=device)
    sin_full = func(data_full)

    keep_size = 7
    # size = int((1 - drop_rate) * num_points)
    keep_ids = sorted(np.random.choice(num_points, size=keep_size, replace=False))
    # keep_ids = sorted(np.random.choice(num_points, size=5, replace=False))
    # keep_ids = [0, 1, 2, 18, 22, 30, 32]

    mask = torch.zeros_like(data_full, dtype=torch.bool, device=device)
    mask[keep_ids] = True
    data_train = data_full[mask].clone()
    # data_train = torch.tensor([-torch.pi/2, torch.pi/2])
    gt_train = func(data_train)

    preds_from_exps = {}
    w2 = {}
    for is_pinn_arg in [True, False]:
        # title = f'Model: sin, len(x_train)={len(data_train)}'
        # fname = f'sin_len_{len(data_train)}'
        # model = SinNN(is_pinn_arg)

        poly_degree = 7
        title = f'Model: Poly {poly_degree} degree, len(t_train)={len(data_train)}'
        fname = f'poly_{poly_degree}_len_{len(data_train)}_plot_pinn_{plot_pinn}'
        model = PolyNN(poly_degree, is_pinn_arg).to(device)

        preds_from_exps[is_pinn_arg], w2[is_pinn_arg] = train(model, data_train, gt_train, data_val, lr, epochs, noise)

    make_animation(data_train, gt_train, data_val, gt_val,
                   w2_arg=w2[True],
                   plot_vanila_arg=plot_vanila,
                   plot_pinn_arg=plot_pinn,
                   preds_pinn_arg=preds_from_exps[True],
                   preds_vanila_arg=preds_from_exps[False],
                   fname_arg=f'm_k/{fname}.gif', fps_arg=fps, title_arg=title, interval_arg=interval)
    print(fname)









