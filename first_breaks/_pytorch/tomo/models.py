from typing import Optional, Union

import torch
from torch import nn
from torch.nn.functional import gelu
import torch.nn.functional as F


class LA_ELU(nn.Module):
    def __init__(self, feat_dim: int, alpha_init: float = 1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, feat_dim) * alpha_init, requires_grad=True)

    def forward(self, x):
        positive = F.elu(x)
        # Negative values are multiplied with the locally adaptive alpha
        negative = self.alpha * (torch.exp(x) - 1) * (x < 0).float()

        return positive + negative


# ACTIVATION_CLASS = nn.Tanh
# ACTIVATION_FUNC = F.tanh

ACTIVATION_CLASS = lambda *args, **kwargs: nn.ELU(alpha=1)
ACTIVATION_FUNC = lambda x: F.elu(x, alpha=1)

# ACTIVATION_CLASS = nn.Mish
# ACTIVATION_FUNC = F.mish

# ACTIVATION_CLASS = nn.GELU
# ACTIVATION_FUNC = F.gelu


class Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, max_value: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim, bias=True)

        # self.activation_class = lambda *args, **kwargs: LA_ELU(hidden_dim)
        # self.activation_func = F.sigmoid

        self.activation_class = ACTIVATION_CLASS
        self.activation_func = ACTIVATION_FUNC

        # layers = [self.activation_class()]
        # for _ in range(num_layers):
        #     layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation_class(), nn.Dropout(0.05)])
        # self.blocks = nn.Sequential(*layers)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.activation_class(), nn.Dropout(0.05)))
        self.blocks = nn.ModuleList(layers)

        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.output_act = nn.Softplus()
        self.max_value = max_value

        for layer in [self.input, *self.blocks, self.output]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=1)
                if layer.bias is not None:
                    layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x):
        x = self.input(x)
        # x = self.blocks(x)
        for block in self.blocks:
            x_new = block(x)
            x = x + x_new
        return self.output_act(self.output(x)) * self.max_value


class Tau(Model):
    def forward(self, source, receiver):
        x = torch.cat([source, receiver], dim=1)
        return super().forward(x)


# class Tau(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, max_value: float):
#         super().__init__()
#         bias = True
#
#         self.input_source = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.input_receiver = nn.Linear(input_dim, hidden_dim, bias=bias)
#
#         self.concat = nn.Linear(2 * hidden_dim, hidden_dim, bias=bias)
#
#         self.activation_class = ACTIVATION_CLASS
#         self.activation_func = ACTIVATION_FUNC
#
#         layers = [self.activation_class()]
#         for _ in range(num_layers):
#             layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=bias), self.activation_class()])
#         self.blocks = nn.Sequential(*layers)
#
#         self.output = nn.Linear(hidden_dim, 1, bias=False)
#         self.output_act = nn.Sigmoid()
#         self.max_value = max_value
#
#         for layer in [self.input_source, self.input_receiver, *self.blocks, self.concat, self.output]:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_normal_(layer.weight.data, gain=1)
#                 if layer.bias is not None:
#                     layer.bias = nn.Parameter(torch.zeros_like(layer.bias))
#
#     def forward(self, source, receiver):
#         source = self.activation_func(self.input_source(source))
#         receiver = self.activation_func(self.input_receiver(receiver))
#         x = self.activation_func(self.concat(torch.cat([source, receiver], dim=1)))
#         x = self.blocks(x)
#         return self.output_act(self.output(x)) * self.max_value


class Eikonal(nn.Module):
    def __init__(self, tau_model, background_velocity: Optional[Union[float, nn.Module]]):
        super().__init__()
        if isinstance(background_velocity, nn.Module):
            self.background_velocity = lambda x: background_velocity(x)
        elif isinstance(background_velocity, (int, float)):
            self.background_velocity = lambda x: background_velocity
        else:
            self.background_velocity = None
        self.tau_model = tau_model

    def forward_t0(self, source, receiver):
        dist = (receiver - source).pow(2).sum(1, keepdim=True).sqrt()
        return dist / self.background_velocity(source)

    def forward(self, source, receiver):
        if self.background_velocity is not None:
            return self.forward_t0(source, receiver) + self.tau_model(source, receiver)
        else:
            return self.tau_model(source, receiver)

    def get_time_and_velocity(self, source, receiver):
        time = self(source, receiver)

        time_r = torch.autograd.grad(
            time,
            receiver,
            grad_outputs=torch.ones_like(time),
            retain_graph=True,
            create_graph=True)[0]

        velocity = (1 / time_r.pow(2).sum(1, keepdim=True)).sqrt()

        return time, velocity

    def get_tensors(self, source, receiver):
        output = {}

        time = self(source, receiver)

        output["time"] = time

        time_r = torch.autograd.grad(
            time,
            receiver,
            grad_outputs=torch.ones_like(time),
            retain_graph=True,
            create_graph=True)[0]

        square_slowness = time_r.pow(2).sum(1, keepdim=True)
        output["square_slowness"] = square_slowness

        square_velocity = 1 / square_slowness
        output["square_velocity"] = square_velocity

        velocity = square_velocity.sqrt()
        output["velocity"] = velocity

        return output


