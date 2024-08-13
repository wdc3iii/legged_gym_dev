import torch
import torch.nn as nn


class ScalarTubeLoss(nn.Module):
    def __init__(self, alpha, delta=1.0):
        super(ScalarTubeLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha = alpha

    def forward(self, fw, w, data):
        residual = w - fw
        loss = torch.where(residual > 0, self.alpha * residual, (1 - self.alpha) * residual.abs())
        return self.huber(loss, torch.zeros_like(loss))


class ScalarHorizonTubeLoss(nn.Module):
    def __init__(self, alpha, delta=1.0):
        super(ScalarHorizonTubeLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.alpha = alpha

    def forward(self, fw, w, data):
        residual = w - fw
        loss = torch.where(residual > 0, self.alpha * residual, (1 - self.alpha) * residual.abs())
        return self.huber(loss, torch.zeros_like(loss))


class VectorTubeLoss(ScalarTubeLoss):
    def __init__(self, alpha, delta=1.0):
        super(VectorTubeLoss, self).__init__(alpha, delta=delta)

    def forward(self, fw, w, data):
        residual = w - fw
        loss = torch.where(residual > 0, self.alpha * residual, (1 - self.alpha) * residual.abs())
        loss = torch.sum(loss, dim=-1)
        return self.huber(loss, torch.zeros_like(loss))


class AlphaScalarTubeLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(AlphaScalarTubeLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, fw, w, data):
        alpha = data[:, -1]
        residual = w - fw
        loss = torch.where(residual > 0, alpha * residual, (1 - alpha) * residual.abs())
        return self.huber(loss, torch.zeros_like(loss))


class AlphaVectorTubeLoss(AlphaScalarTubeLoss):
    def __init__(self, delta=1.0):
        super(AlphaVectorTubeLoss, self).__init__(delta=delta)

    def forward(self, fw, w, data):
        alpha = data[:, -1]
        residual = w - fw
        loss = torch.where(residual > 0, alpha * residual, (1 - alpha) * residual.abs())
        loss = torch.sum(loss, dim=-1)
        return self.huber(loss, torch.zeros_like(loss))


class ErrorLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fe, e, data):
        return self.mse(fe, e)
