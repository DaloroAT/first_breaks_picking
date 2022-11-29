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


def loss_classic(model, data, gt):
    preds = model(data)
    loss = torch.sum(torch.pow(gt - preds, 2))
    return loss


def loss_pinn(model, data, gt):
    preds = model(data)

    y_x = torch.autograd.grad(preds, data, grad_outputs=torch.ones_like(preds), retain_graph=True, create_graph=True)[0]
    y_xx = torch.autograd.grad(y_x, data, grad_outputs=torch.ones_like(y_x), retain_graph=True, create_graph=True)[0]
    res1 = y_xx + preds
    loss = torch.sum(torch.pow(gt - preds, 2)) + torch.sum(torch.pow(res1, 2))
    # loss = torch.sum(torch.abs(gt - preds)) + torch.sum(torch.abs(res1))

    return loss


class PolyNN(nn.Module):
    def __init__(self, polynomial_degree=5):
        super().__init__()
        self.polynomial_degree = polynomial_degree
        self.coeffs = nn.Linear(polynomial_degree - 1, 1, bias=True)
        self.coeffs.bias.data = torch.zeros_like(self.coeffs.bias.data)
        self.init_with_const(0)

    def init_with_const(self, const=0):
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


def make_animation(data_train, gt_train, data_val, gt_val, preds_pinn, preds_vanila, fname, fps=10, title=''):
    fig = plt.figure(figsize=(16, 9))
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlim([min(data_val), max(data_val)])
    axes.set_ylim([min(gt_val) - 0.5, max(gt_val) + 0.5])

    plt.xlabel("X")
    plt.ylabel("Y")

    t = plt.title(title, fontdict={'size': 14})

    plt.plot(data_val, gt_val, color=(0, 0, 0, 0.3), linewidth=3, label='sin(x)')
    plt.scatter(data_train, gt_train, color=(3 / 256, 125 / 256, 80 / 256), linewidth=4, label='x_train')

    pinn_line = plt.plot(data_val, [None] * len(data_val), color='r', label='PINN', linewidth=3)[0]
    vanila_line = plt.plot(data_val, [None] * len(data_val), color='b', label='Vanila', linewidth=3)[0]

    plt.legend(loc='upper right', prop={'size': 14})

    fig.tight_layout()

    pbar = tqdm('Saving animation', total=len(preds_pinn))

    def animation(step):
        step_title = (title + f" , epoch={step}").strip(' , ')
        t.set_text(step_title)
        pinn_line.set_ydata(preds_pinn[step, :])
        vanila_line.set_ydata(preds_vanila[step, :])
        assert step < len(preds_pinn)
        pbar.update(1)

    animation = FuncAnimation(fig, animation, frames=len(preds_pinn))
    animation.save(fname, fps=fps)
    pbar.close()


def func(dataset):
    # return torch.sin(dataset * torch.pi)
    return torch.sin(dataset)


def train(model, data_train_arg, gt_train_arg, data_val_arg, lr, is_pinn, epochs, noise=0):
    optimizer = Adam(lr=lr, params=model.parameters())

    pbar = tqdm(range(epochs))

    extra_noise = 0.2 * torch.randn(data_train_arg.shape)

    full_preds_val = torch.zeros((epochs, len(data_val_arg)))

    for epoch in pbar:
        optimizer.zero_grad()
        data = data_train_arg.clone().detach().requires_grad_(True)[:, None]
        gt = (gt_train_arg + extra_noise).clone().detach().requires_grad_(True)[:, None]

        if is_pinn:
            loss = loss_pinn(model, data, gt)
        else:
            loss = loss_classic(model, data, gt)

        loss.backward()
        optimizer.step()

        pbar.set_postfix_str(loss.item())

        with torch.no_grad():
            data_val = data_val_arg.clone().detach().requires_grad_(True)[:, None]

            preds_val = model(data_val)
            full_preds_val[epoch, :] = preds_val.squeeze()

    return full_preds_val


if __name__ == "__main__":
    set_global_seed(1)

    lr = 3e-2
    num_points = 40
    drop_rate = 0.0
    epochs = 300
    noise = 0
    fps = 20

    diap = [-torch.pi, torch.pi]

    data_val = torch.linspace(*diap, 1000)
    gt_val = func(data_val)
    data_full = torch.linspace(*diap, num_points)
    sin_full = func(data_full)

    keep_ids = sorted(np.random.choice(num_points, size=int((1 - drop_rate) * num_points), replace=False))
    # keep_ids = sorted(np.random.choice(num_points, size=5, replace=False))
    # keep_ids = [0, 1, 2, 18, 22, 30, 32]

    mask = torch.zeros_like(data_full, dtype=torch.bool)
    mask[keep_ids] = True
    data_train = data_full[mask].clone()
    # data_train = torch.tensor([-torch.pi/2, torch.pi/2])
    gt_train = func(data_train)

    preds_from_exps = {}
    for is_pinn in [True, False]:
        # degree = 5
        # title = f'Model: Poly {degree} degree, len(x_train)={len(data_train)}'
        # fname = f'poly_{degree}_len_{len(data_train)}'
        # model = PolyNN(degree)

        degree = 5
        title = f'Model: sin, len(x_train)={len(data_train)}'
        fname = f'sin_len_{len(data_train)}'
        model = SinNN()

        preds_from_exps[is_pinn] = train(model, data_train, gt_train, data_val, lr, is_pinn, epochs, noise)

    make_animation(data_train, gt_train, data_val, gt_val,
                   preds_pinn=preds_from_exps[True],
                   preds_vanila=preds_from_exps[False],
                   fname=f'{fname}.gif', fps=fps, title=title)
    print(fname)









