import torch


x = torch.tensor([-3, -2, -1, 0, 1, 2, 3],
                 dtype=torch.float,
                 requires_grad=True).view(-1, 1)
y = x ** 2

y_x = torch.autograd.grad(
    y,
    x,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True)[0]

y_xx = torch.autograd.grad(
    y_x,
    x,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True)[0]

print(torch.cat([x, y_x, y_xx], dim=1).int())


x1 = torch.tensor(
    [-2, -1, 0, 1, 2],
    dtype=torch.float,
    requires_grad=True).view(-1, 1)
x2 = torch.tensor(
    [-2, -1, 0, 1, 2],
    dtype=torch.float,
    requires_grad=True).view(-1, 1)

y = x1 ** 2 * x2 ** 3

y_x1 = torch.autograd.grad(
    y,
    x1,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True)[0]

y_x2 = torch.autograd.grad(
    y,
    x2,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True)[0]

y_x1x2 = torch.autograd.grad(
    y_x1,
    x2,
    grad_outputs=torch.ones_like(y),
    retain_graph=True,
    create_graph=True)[0]

print(torch.cat([x1, y_x1, y_x2, y_x1x2], dim=1).int())




