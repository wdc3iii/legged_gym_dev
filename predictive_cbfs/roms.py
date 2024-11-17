import torch

class SingleInt2D:

    def __init__(self, dt):
        self.dt = dt
        self.n = 2
        self.m = 2

    def f(self, z):
        return torch.zeros_like(z).to(torch.device("cuda:0"))

    def g(self, z):
        return torch.eye(2).expand(*z.shape[:-1], 2, 2).to(torch.device("cuda:0"))

    def __call__(self, z, v):
        return z + v * self.dt

    def proj(self, x):
        return x[..., :2]


class DoubleInt2D:

    def __init__(self, dt):
        self.dt = dt
        self.n = 4
        self.m = 2

    def f(self, z):
        return torch.zeros_like(z).to(torch.device("cuda:0"))

    def g(self, z):
        g = torch.zeros(z.shape[0], 4, 2)
        g[:, 2, 0] = 1
        g[:, 3, 1] = 1
        return g.to(torch.device("cuda:0"))

    def __call__(self, z, v):
        return z + torch.cat((z[..., 2:], v), dim=-1) * self.dt

    def proj(self, x):
        return x[..., :4]


class SingleInt1D:

    def __init__(self, dt):
        self.dt = dt
        self.n = 1
        self.m = 1

    def f(self, z):
        return torch.zeros_like(z).to(torch.device("cuda:0"))

    def g(self, z):
        return torch.ones_like(z)[..., None].to(torch.device("cuda:0"))

    def __call__(self, z, v):
        return z + v * self.dt

    def proj(self, x):
        return x[..., 0, None]


class DoubleInt1D:

    def __init__(self, dt):
        self.dt = dt
        self.n = 2
        self.m = 1

    def f(self, z):
        return torch.zeros_like(z).to(torch.device("cuda:0"))

    def g(self, z):
        g = torch.zeros_like(z).to(torch.device("cuda:0"))[:, None]
        g[..., 1, 0] = 1
        return g

    def __call__(self, z, v):
        return z + torch.cat((z[..., 1:], v), dim=-1) * self.dt

    def proj(self, x):
        return x[..., :2]