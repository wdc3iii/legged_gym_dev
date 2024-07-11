import torch
import torch.nn as nn


class ScalarTubeLoss(nn.Module):
    def __init__(self, alpha, delta=1.0):
        super(ScalarTubeLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha = alpha

    def forward(self, fw, w, data):
        residual = w - fw
        # TODO: Check with Cole direction of residual inequality
        loss = torch.where(residual > 0, self.alpha * residual, (1 - self.alpha) * residual.abs())
        return self.huber(loss, torch.zeros_like(loss))


class VectorTubeLoss(ScalarTubeLoss):
    def __init__(self, alpha, delta=1.0):
        super(VectorTubeLoss, self).__init__(alpha, delta=delta)

    def forward(self, fw, w, data):
        w = w.squeeze(1)
        residual_x = w[:, 0] - fw[:, 0]
        residual_y = w[:, 1] - fw[:, 1]
        norm_residual = torch.sqrt(residual_x**2 + residual_y**2)
        loss = torch.where(norm_residual <= 0, self.alpha * norm_residual, (1 - self.alpha) * norm_residual.abs())
        return self.huber(loss, torch.zeros_like(loss))


class AlphaScalarTubeLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(AlphaScalarTubeLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, fw, w, alpha):
        residual = w - fw
        loss = torch.where(residual <= 0, alpha * residual, (1 - alpha) * residual.abs())
        return self.huber(loss, torch.zeros_like(loss))


class AlphaVectorTubeLoss(AlphaScalarTubeLoss):
    def __init__(self, delta=1.0):
        super(AlphaVectorTubeLoss, self).__init__(delta=delta)

    def forward(self, fw, w, alpha):
        w = w.squeeze(1)
        residual_x = w[:, 0] - fw[:, 0]
        residual_y = w[:, 1] - fw[:, 1]
        norm_residual = torch.sqrt(residual_x**2 + residual_y**2)
        loss = torch.where(norm_residual <= 0, alpha * norm_residual, (1 - alpha) * norm_residual.abs())
        return self.huber(loss, torch.zeros_like(loss))
