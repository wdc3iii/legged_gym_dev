import torch
from torch import nn

class CheckLoss(nn.Module):
    def __init__(self, alpha, delta=1):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, delta, delta_targ):
        # Want delta > delta_targ. Therefore, choose  0 << alpha < 1, penalize targ - delta > 0 heavily
        e = delta_targ - delta
        loss = torch.where(e >= 0, self.alpha * e, (1 - self.alpha) * e.abs())
        return self.huber(loss, torch.zeros_like(loss))

class SquaredLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, delta, delta_targ):
        return self.mse(delta, delta_targ)