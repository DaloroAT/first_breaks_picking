import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation, dropout=0.0, use_norm_layer=False, use_skip=True):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.use_skip = use_skip
        self.use_norm_layer = use_norm_layer
        self.dropout = dropout

        self.input_layer = nn.Linear(self.in_dim, self.hidden_dim)
        self.inner_blocks = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.out_dim = nn.Linear(self.hidden_dim, self.out_dim)
        if self.use_norm_layer:
            self.norm_layer = nn.LayerNorm(self.hidden_dim)
        else:
            self.norm_layer = nn.Identity()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for block in self.inner_blocks:
            if self.use_skip:
                x = x + self.activation(block(x))
            else:
                x = self.activation(block(x))

            x = self.norm_layer(x)
            x = self.dropout_layer(x)

        return x


class TauNew(nn.Module):
    def __init__(self, dim=2, hidden_dim=20, num_inner_layers=2, num_blocks=5, dropout=0.0):
        super(TauNew, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_inner_layers = num_inner_layers
        self.num_blocks = num_blocks
        self.act = nn.Tanhshrink()
        self.dropout = dropout

        mlp_kwargs = {"in_dim": self.dim,
                      "out_dim": self.hidden_dim,
                      "hidden_dim": self.hidden_dim,
                      "num_layers": self.num_inner_layers,
                      "activation": self.act,
                      "use_skip": True,
                      "use_norm_layer": False}

        concat_kwargs = {"in_dim": 2 * self.hidden_dim,
                         "out_dim": self.hidden_dim,
                         "hidden_dim": self.hidden_dim,
                         "num_layers": self.num_inner_layers,
                         "activation": self.act,
                         "use_skip": True,
                         "use_norm_layer": True}

        block_kwargs = {"in_dim": self.hidden_dim,
                        "out_dim": self.hidden_dim,
                        "hidden_dim": self.hidden_dim,
                        "num_layers": self.num_blocks,
                        "activation": self.act,
                        "use_skip": True,
                        "use_norm_layer": True,
                        "dropout": self.dropout}

        self.input_source = MLP(**mlp_kwargs)
        self.input_receiver = MLP(**mlp_kwargs)
        self.concat = MLP(**concat_kwargs)
        self.blocks = MLP(**block_kwargs)
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, source, receiver):
        source = self.input_source(source)
        receiver = self.input_receiver(receiver)

        features = torch.cat([source, receiver], dim=1)
        features = self.concat(features)
        features = self.blocks(features)

        features = self.output(features)

        return features


class Tau(nn.Module):
    def __init__(self, dim=2, hidden_size=20, num_layers=5):
        super(Tau, self).__init__()
        self.dim = dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        # self.act = torch.atan
        # self.act = torch.sin
        # self.act = nn.ReLU()
        # self.act = nn.LeakyReLU()
        # self.act = nn.Tanhshrink()
        self.act = nn.Tanh()

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
            features = (features + self.act(block(features)))
            # features = self.act(block(features))

        features = self.act(features)

        features = self.output(features)
        # features = torch.log1p(features)
        return features
