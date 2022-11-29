from math import factorial
from operator import itemgetter

import cv2
import torch
from oml.utils.misc import set_global_seed
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from tqdm import tqdm


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
    def __init__(self, polynomial_degree):
        super().__init__()
        self.polynomial_degree = polynomial_degree
        self.coeffs = nn.Linear(polynomial_degree - 1, 1, bias=True)
        self.coeffs.bias.data = torch.zeros_like(self.coeffs.bias.data)

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


def func(dataset):
    # return torch.sin(dataset * torch.pi)
    return torch.sin(dataset)


set_global_seed(1)
degree = 7
lr = 4e-2
num_points = 40
drop_rate = 0
epochs = 700
is_pinn = False

diap = [-torch.pi, torch.pi]

data_back = torch.linspace(*diap, 1000)
sin_back = func(data_back)
data_full = torch.linspace(*diap, num_points)
sin_full = func(data_full)

keep_ids = sorted(np.random.choice(num_points, size=int((1 - drop_rate) * num_points), replace=False))
# keep_ids = sorted(np.random.choice(num_points, size=5, replace=False))
# keep_ids = [0, 1, 2, 18, 22, 30, 32]

mask = torch.zeros_like(data_full, dtype=torch.bool)
mask[keep_ids] = True
data_dropped = data_full[mask].clone()
# data_dropped = torch.tensor([-torch.pi, -torch.pi / 2, torch.pi / 2])

# model = PolyNN(degree)
# model.init_with_const(0)

model = SinNN()
optimizer = Adam(lr=lr, params=model.parameters())

#
#
# plt.plot(data_back, sin_back, color=(0, 0, 0, 0.3))
# with torch.no_grad():
#     plt.plot(data_back, model(data_back[:, None]).squeeze(), color='b')
# plt.scatter(data_dropped, torch.sin(data_dropped), color='y')
# plt.xlim(diap)
# plt.ylim([-2, 2])
#
# plt.show()


out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))


# print(model.coeffs.weight.data)

pbar = tqdm(range(epochs))

# noise = 0.2 * torch.randn(data_dropped.shape)
noise = 0

for _ in pbar:
    optimizer.zero_grad()
    data = data_dropped.clone().detach().requires_grad_(True)[:, None]
    gt = (func(data_dropped) + noise).clone().detach().requires_grad_(True)[:, None]

    if is_pinn:
        loss = loss_pinn(model, data, gt)
    else:
        loss = loss_classic(model, data, gt)

    # if not is_pinn:
    #     preds = model(data)
    #     # loss = torch.sum(torch.abs(gt - preds))
    #     loss = torch.mean(torch.pow(gt - preds, 2))
    # else:
    #     preds = model(data)
    #     res = model.residual(data)
    #     loss = torch.mean(torch.pow(gt - preds, 2)) + torch.mean(torch.pow(res, 2))

    loss.backward()
    optimizer.step()

    pbar.set_postfix_str(loss.item())

    with torch.no_grad():
        plt.plot(data_back, sin_back, color=(0, 0, 0, 0.3))
        plt.scatter(data_dropped, func(data_dropped) + noise, color='y')
        plt.xlim(diap)
        plt.ylim([-2, 2])

        data_val = data_back.clone().detach().requires_grad_(True)[:, None]
        preds_val = model(data_val)
        plt.plot(data_val.squeeze(), preds_val.squeeze(), color='r')

        title = 'pinn' if is_pinn else 'vanila'

        plt.gcf().canvas.draw()
        image_from_plot = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        out.write(image_from_plot)

        plt.close()

    # assert not torch.any(torch.isnan(model.coeffs.weight.data)), 'Divergence!'


out.release()

# print(model.coeffs.weight.data)

# print(model.coeffs.weight.data)









