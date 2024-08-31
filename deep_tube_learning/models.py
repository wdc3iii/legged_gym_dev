import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size, output_dim, num_units, num_layers, activation, final_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, num_units), activation])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_units, num_units))
            self.layers.append(activation)
        self.layers.append(nn.Linear(num_units, output_dim))
        if final_activation is not None:
            self.layers.append(final_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RecursiveMLP(nn.Module):
    def __init__(self, input_size, output_dim, num_units, num_layers, activation, final_activation=None, stop_grad=True):
        super(RecursiveMLP, self).__init__()
        # Compute Hrev, Hfwd from input_size, output_dim
        # TODO size properly
        self.n = 2
        self.m = 2
        self.H_fwd = output_dim
        self.H_rev = (input_size - (self.n - 2) - self.H_fwd * self.m) / (1 + self.m)
        assert self.H_rev == int(self.H_rev)
        self.H_rev = int(self.H_rev)
        in_size = self.H_rev + (self.n - 2) + self.H_rev * self.m + 1
        self.mlp = MLP(in_size, 1, num_units, num_layers, activation, final_activation)
        self.stop_grad = stop_grad

    def forward(self, x):
        t = torch.arange(self.H_fwd, device=x.device)
        w = torch.zeros((x.shape[0], self.H_fwd), device=x.device)
        e = x[:, :self.H_rev]
        v = x[:, -(self.H_fwd + self.H_rev) * self.m:].reshape((x.shape[0], self.H_fwd + self.H_rev, self.m))
        for i in range(self.H_fwd):
            if i < self.H_rev:
                w_seg = torch.clone(w[:, 0:i])
                if self.stop_grad:
                    w_seg = w_seg.detach()
                data = torch.concatenate([
                    e[:, i:], w_seg,
                    v[:, i:i + self.H_rev].reshape((x.shape[0], -1)),
                    t[i] * torch.ones((x.shape[0], 1), device=x.device)
                ], dim=1)
            else:
                w_seg = torch.clone(w[:, i - self.H_rev:i])
                if self.stop_grad:
                    w_seg = w_seg.detach()
                data = torch.concatenate([
                    w_seg,
                    v[:, i:i + self.H_rev].reshape((x.shape[0], -1)),
                    t[i] * torch.ones((x.shape[0], 1), device=x.device)
                ], dim=1)
            w[:, i] = self.mlp.forward(data)[:].squeeze(dim=1)
        return w
