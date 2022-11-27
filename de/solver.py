import torch
from torch import nn


class Model(nn.Module):
    def forward(self, x):
        return x ** 3


x = torch.tensor([2.0], requires_grad=True)[None, :]
model = Model()

y = model(x)

y_x = torch.autograd.grad(
    y, x,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True
)[0]

print(model(x))
print(y_x)





