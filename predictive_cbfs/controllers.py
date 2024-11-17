import torch

class RobustifiedCBF:

    def __init__(self, cbf, nom_controller, delta, dyn):
        self.cbf = cbf
        self.vd = nom_controller
        self.delta = delta.to(torch.device("cuda:0"))
        self.dyn = dyn
        self.delta_val = None

    def filtered_v(self, x):
        z: torch.Tensor = self.dyn.proj(x)
        vd: torch.Tensor= self.vd(z)
        self.delta_val = torch.clip(self.delta(x).squeeze(-1).detach(), min=0)
        h, Jh = self.cbf.h_Jh(z)
        a: torch.Tensor = -self.cbf.alpha * h - torch.sum(Jh * self.dyn.f(z), dim=-1) + self.delta_val + self.cbf.epsilon
        b: torch.Tensor = torch.matmul(self.dyn.g(z), Jh.unsqueeze(-1)).squeeze(-1)
        b_vd = torch.sum(b * vd, dim=-1)
        return torch.where(torch.greater_equal(b_vd, a)[:, None], vd, vd + b / torch.sum(b * b, dim=-1, keepdim=True) * (a - b_vd)[:, None])

    def __call__(self, x):
        return self.filtered_v(x)



class HopperVelTracking:

    def __init__(self, raibert_heuristic, robustified_cbf, t_prop):
        self.rh = raibert_heuristic.get_inference_policy(None)
        self.v_filt = robustified_cbf
        self.t_prop = t_prop

    def __call__(self, obs):
        v = self.v_filt(obs)
        obs[..., :2] = v * self.t_prop
        obs[..., 4:] = v
        return self.rh(obs)

class DoubleSingleVelTracking:
    def __init__(self, k_d, robustified_cbf):
        self.k_d = k_d
        self.v_filt = robustified_cbf

    def __call__(self, obs):
        v = self.v_filt(obs)
        n = obs.shape[-1]
        return -self.k_d * (obs[..., n//2:] - v)


class SingleIntPosTracking:

    def __init__(self, k, zd, v_max):
        self.k = k
        self.zd = torch.tensor(zd, device="cuda:0")
        self.v_max = v_max

    def __call__(self, z):
        vd = -self.k * (z - self.zd)
        norm_vd = torch.linalg.norm(vd, dim=-1, keepdim=True)
        return torch.where(torch.greater(norm_vd, self.v_max), vd / norm_vd * self.v_max, vd)


class SingleIntConstVel:

    def __init__(self, v):
        self.v = torch.tensor([v])

    def __call__(self, z):
        return self.v[None, :].repeat(z.shape[0], 1).to(z.device)
