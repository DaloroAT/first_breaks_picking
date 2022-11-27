import torch
from oml.utils.misc import set_global_seed
from torch import nn
from torch.optim import Adam
from torchvision import models
from tqdm import tqdm


class Solver(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        input_dim = dimension
        hidden_dim = 10
        hidden_layers = 10

        # self.act = nn.Tanh()
        self.act = nn.ELU()

        self.r0_module = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act)
        self.r1_module = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.act)
        self.inp_module = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), self.act)

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])

        self.t = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, r0, r1):
        r0 = self.r0_module(r0)
        r1 = self.r1_module(r1)

        features = self.inp_module(torch.cat([r0, r1], dim=1))

        for layer in self.hidden_layers:
            features = features + self.act(layer(features))

        return self.t(features)


class TravelTime(nn.Module):
    def __init__(self, solver: Solver):
        super().__init__()
        self.solver = solver

    def forward(self, r0, r1):
        return self.solver(r0, r1)

    def slowness(self, r0, r1):
        # Xp = torch.cat([r0, r1], dim=1)
        # tau = self(Xp[:, :3], Xp[:, 3:])
        #
        # dtau = torch.autograd.grad(outputs=tau, inputs=Xp, grad_outputs=torch.ones_like(tau),
        #                     only_inputs=True,create_graph=True,retain_graph=True)[0]
        #
        # T0    = torch.sqrt(((Xp[:,3]-Xp[:,0])**2 + (Xp[:,4]-Xp[:,1])**2 + (Xp[:,5]-Xp[:,2])**2))
        # T1    = (T0**2)*(dtau[:,3]**2 + dtau[:,4]**2 + dtau[:,5]**2)
        # T2    = 2*tau[:,0]*(dtau[:,3]*(Xp[:,3]-Xp[:,0]) + dtau[:,4]*(Xp[:,4]-Xp[:,1]) + dtau[:,5]*(Xp[:,5]-Xp[:,2]))
        # T3    = tau[:,0]**2
        # S2    = (T1+T2+T3)
        # return S2

        t = self(r0, r1)

        dt = torch.autograd.grad(
            t, r1,
            grad_outputs=torch.ones_like(t),
            retain_graph=True,
            create_graph=True
        )[0]

        return torch.sqrt(torch.sum(torch.pow(dt, 2), dim=1))


if __name__ == "__main__":
    set_global_seed(1)
    device = 'cuda'
    dimension = 3
    lr = 5e-4
    num_epochs = 600
    bsize = 100000
    slowness = 0.1

    tt = TravelTime(Solver(dimension)).to(device)

    optim = Adam(lr=lr, params=tt.parameters())

    pbar = tqdm(range(num_epochs))

    for _ in pbar:
        optim.zero_grad()
        source = torch.randn((bsize, dimension), requires_grad=True, dtype=torch.float, device=device)
        rec = torch.randn((bsize, dimension), requires_grad=True, dtype=torch.float, device=device)
        s_pred = tt.slowness(source, rec)
        s_gt = torch.tensor(slowness, device=device).repeat(len(s_pred))

        loss = torch.mean(torch.pow(s_pred - s_gt, 2))
        loss.backward()
        optim.step()

        pbar.set_postfix_str(loss.item())

    # tt.eval()
    with torch.no_grad():
        r0 = torch.tensor([[0, 0, 0]], requires_grad=True, dtype=torch.float, device=device)
        r1 = torch.tensor([[1, 0, 0]], requires_grad=True, dtype=torch.float, device=device)
    print(tt.slowness(r0, r1))
    print(tt(r0, r1))



    # print(tt.square_slowness(source, rec))

    # print(tt(data).shape)



