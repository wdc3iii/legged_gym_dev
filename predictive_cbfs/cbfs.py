from abc import ABC, abstractmethod
import torch

class CBF(ABC):

    @abstractmethod
    def h(self, z):
        raise NotImplementedError

    @abstractmethod
    def h_Jh(self, z):
        raise NotImplementedError


class ObstacleCBF(CBF):

    def __init__(self, alpha, epsilon, rs, cxs, cys):
        self.alpha = alpha
        self.epsilon = epsilon
        self.rs = torch.tensor(rs, device="cuda:0")
        self.cs = torch.vstack((torch.tensor(cxs), torch.tensor(cys))).to(torch.device("cuda:0"))

    def _p_center(self, z):
        """
        Computes the displacement from the center of each obstacle
        """
        return z.unsqueeze(1) - self.cs.unsqueeze(0)

    def _d(self, z):
        """
        Computes the distance to each obstacle
        """
        return torch.linalg.norm(self._p_center(z), dim=-1) - self.rs.unsqueeze(0)

    def h(self, z):
        """
        Compute the value of the cbf
        :param z: [..., 2] array of positions
        """
        return torch.min(self._d(z), dim=-1)[0]

    def h_Jh(self, z):
        """
        Compute the value and Jacobian of the cbf
        :param z: [..., 2] array of positions
        """
        pc = self._p_center(z)
        norm_pc = torch.linalg.norm(pc, dim=-1)
        d = norm_pc - self.rs.unsqueeze(0)
        min_d, min_inds = torch.min(d, dim=-1)
        # Indexing the correct elements of pc and norm_pc
        min_pc = pc[torch.arange(pc.size(0)), min_inds]  # Shape: Nx2
        min_norm_pc = norm_pc[torch.arange(norm_pc.size(0)), min_inds]  # Shape: N
        return min_d, min_pc / min_norm_pc[:, None]


class SmoothObstacleCBF(ObstacleCBF):
    def __init__(self, alpha, epsilon, rs, cxs, cys, rho):
        super().__init__(alpha, epsilon, rs, cxs, cys)
        self.rho = rho

    def h_sum_exp(self, d):
        return torch.sum(torch.exp(-self.rho * d), dim=-1)

    def h_log_of_sum_exp(self, sum_exp):
        return - 1 / self.rho * torch.log(sum_exp)

    def h(self, z):
        """
        Compute the value of the cbf
        :param z: [..., 2] array of positions
        """
        return self.h_log_of_sum_exp(self.h_sum_exp(self._d(z)))

    def h_Jh(self, z):
        """
        Compute the value and Jacobian of the cbf
        :param z: [..., 2] array of positions
        """
        pc = self._p_center(z)
        norm_pc = torch.linalg.norm(pc, dim=-1)
        d = norm_pc - self.rs.unsqueeze(0)
        exp_d = torch.exp(-self.rho * d)
        sum_exp = torch.sum(exp_d, dim=-1)

        return self.h_log_of_sum_exp(sum_exp), 1 / sum_exp * torch.sum(pc / norm_pc[..., None] * exp_d, dim=-1)


class ZeroCBF(CBF):

    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

    def h(self, z):
        return z.squeeze(-1)

    def h_Jh(self, z):
        return z.squeeze(-1), torch.ones_like(z)