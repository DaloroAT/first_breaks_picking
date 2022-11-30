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


def sin_coeffs(polynomial_degree):
    coeffs = []

    for n in range(polynomial_degree):
        if n % 2 == 0:
            coeffs.append(0)
        else:
            coeffs.append((-1) ** ((n - 1) / 2) / factorial(n))

    return coeffs


def loss_classic(model_arg, data_arg, gt_arg):
    preds = model_arg(data_arg)
    loss = torch.sum(torch.pow(gt_arg - preds, 2))
    return loss


def loss_pinn(model_arg, data_arg, gt_arg):
    preds = model_arg(data_arg)

    y_x = torch.autograd.grad(preds, data_arg, grad_outputs=torch.ones_like(preds), retain_graph=True, create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, data_arg, grad_outputs=torch.ones_like(y_x), retain_graph=True, create_graph=True)[0]
    res1 = y_xx + preds
    loss = torch.sum(torch.pow(gt_arg - preds, 2)) + torch.sum(torch.pow(res1, 2))
    # loss = torch.sum(torch.abs(gt - preds)) + torch.sum(torch.abs(res1))

    return loss


class PolyNN(nn.Module):
    def __init__(self, polynomial_degree=5):
        super().__init__()
        self.polynomial_degree = polynomial_degree
        self.coeffs = nn.Linear(polynomial_degree - 1, 1, bias=True)

        self.coeffs.bias.data = torch.zeros_like(self.coeffs.bias.data)
        self.init_with_const(0.1)

    def init_with_const(self, const=0.0):
        self.coeffs.weight.data = (const * torch.ones(self.polynomial_degree - 1)).requires_grad_(True)[None, :]

    def init_with_exact(self):
        before = self.coeffs.weight.data.shape
        self.coeffs.weight.data = torch.tensor(sin_coeffs(self.polynomial_degree)[1:], requires_grad=True)[None, :]
        after = self.coeffs.weight.data.shape
        assert before == after, [before, after]

    def forward(self, x):
        features = []
        for degree in range(1, self.polynomial_degree):
            features.append(x ** degree)
        features = torch.cat(features, dim=1)

        return self.coeffs(features)


class SinNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.amp = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.y_shift = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.x_shift = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.freq = nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def forward(self, x):
        return self.y_shift + self.amp * torch.sin(self.freq * (x - self.x_shift))


def make_animation(data_train_arg, gt_train_arg, data_val_arg, gt_val_arg, preds_pinn_arg,
                   preds_vanila_arg, fname_arg, fps_arg=10, title_arg='', interval_arg=1):
    fig = plt.figure(figsize=(16, 9))
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlim([min(data_val_arg), max(data_val_arg)])
    axes.set_ylim([min(gt_val_arg) - 0.5, max(gt_val_arg) + 0.5])

    plt.xlabel("X")
    plt.ylabel("Y")

    t = plt.title(title_arg, fontdict={'size': 14})

    plt.plot(data_val_arg, gt_val_arg, color=(0, 0, 0, 0.3), linewidth=3, label='sin(x)')
    plt.scatter(data_train_arg, gt_train_arg, color=(3 / 256, 125 / 256, 80 / 256), linewidth=4, label='x_train')

    pinn_line = plt.plot(data_val_arg, [None] * len(data_val_arg), color='r', label='PINN', linewidth=3)[0]
    vanila_line = plt.plot(data_val_arg, [None] * len(data_val_arg), color='b', label='Vanila', linewidth=3)[0]

    plt.legend(loc='upper right', prop={'size': 14})

    fig.tight_layout()

    frames = len(preds_pinn_arg) // interval_arg

    pbar = tqdm('Saving animation', total=frames)

    def animation(step):
        step = step * interval_arg
        step_title = (title_arg + f" , epoch={step}").strip(' , ')
        t.set_text(step_title)
        pinn_line.set_ydata(preds_pinn_arg[step, :])
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
          lr_arg, is_pinn_arg, epochs_arg, noise_arg=0):
    optimizer = Adam(lr=lr_arg, params=model_arg.parameters())

    pbar = tqdm(range(epochs_arg))

    extra_noise = noise_arg * torch.randn(data_train_arg.shape)

    full_preds_val = torch.zeros((epochs_arg, len(data_val_arg)))

    for epoch in pbar:
        optimizer.zero_grad()
        data = data_train_arg.clone().detach().requires_grad_(True)[:, None]
        gt = (gt_train_arg + extra_noise).clone().detach().requires_grad_(True)[:, None]

        if is_pinn_arg:
            loss = loss_pinn(model_arg, data, gt)
        else:
            loss = loss_classic(model_arg, data, gt)

        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(loss.item())

        with torch.no_grad():
            data_val_epoch = data_val_arg.clone().detach().requires_grad_(True)[:, None]

            preds_val = model_arg(data_val_epoch)
            full_preds_val[epoch, :] = preds_val.squeeze()

    return full_preds_val


if __name__ == "__main__":
    set_global_seed(1)

    lr = 4e-2
    num_points = 40
    drop_rate = 0.85
    epochs = 3200
    noise = 0
    fps = 20
    interval = 20

    diap = [-torch.pi, torch.pi]

    data_val = torch.linspace(*diap, 1000)
    gt_val = func(data_val)
    data_full = torch.linspace(*diap, num_points)
    sin_full = func(data_full)

    keep_size = 40
    # size = int((1 - drop_rate) * num_points)
    keep_ids = sorted(np.random.choice(num_points, size=keep_size, replace=False))
    # keep_ids = sorted(np.random.choice(num_points, size=5, replace=False))
    # keep_ids = [0, 1, 2, 18, 22, 30, 32]

    mask = torch.zeros_like(data_full, dtype=torch.bool)
    mask[keep_ids] = True
    data_train = data_full[mask].clone()
    # data_train = torch.tensor([-torch.pi/2, torch.pi/2])
    gt_train = func(data_train)

    preds_from_exps = {}
    for is_pinn in [True, False]:
        degree = 7
        title = f'Model: Poly {degree} degree, len(x_train)={len(data_train)}'
        fname = f'poly_{degree}_len_{len(data_train)}'
        model = PolyNN(degree)

        # degree = 5
        # title = f'Model: sin, len(x_train)={len(data_train)}'
        # fname = f'sin_len_{len(data_train)}'
        # model = SinNN()

        preds_from_exps[is_pinn] = train(model, data_train, gt_train, data_val, lr, is_pinn, epochs, noise)

        if isinstance(model, PolyNN):
            print(model.coeffs.bias.data, model.coeffs.weight.data)

    make_animation(data_train, gt_train, data_val, gt_val,
                   preds_pinn_arg=preds_from_exps[True],
                   preds_vanila_arg=preds_from_exps[False],
                   fname_arg=f'gifs/{fname}_print.gif', fps_arg=fps, title_arg=title, interval_arg=interval)
    print(fname)









