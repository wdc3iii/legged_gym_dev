import torch

class RobustifiedCBF:

    def __init__(self, cbf, nom_controller, delta, dyn, delta_max):
        self.cbf = cbf
        self.vd = nom_controller
        self.delta = delta.to(torch.device("cuda:0"))
        self.dyn = dyn
        self.delta_val = None
        self.delta_max = delta_max

    def filtered_v(self, x):
        z: torch.Tensor = self.dyn.proj(x)
        vd: torch.Tensor= self.vd(z)
        self.delta_val = torch.clip(self.delta(x).squeeze(-1).detach(), min=0, max=self.delta_max)
        h, Jh = self.cbf.h_Jh(z)
        a: torch.Tensor = -self.cbf.alpha * h - torch.sum(Jh * self.dyn.f(z), dim=-1) + self.delta_val + self.cbf.epsilon
        b: torch.Tensor = torch.matmul(self.dyn.g(z), Jh.unsqueeze(-1)).squeeze(-1)
        b_vd = torch.sum(b * vd, dim=-1)
        return torch.where(torch.greater_equal(b_vd, a)[:, None], vd, vd + b / torch.sum(b * b, dim=-1, keepdim=True) * (a - b_vd)[:, None])

    def __call__(self, x):
        return self.filtered_v(x)


class HopperVelTracking:

    def __init__(self, raibert_heuristic, robustified_cbf, t_prop, keep_inds=None):
        self.rh = raibert_heuristic.get_inference_policy(None)
        self.v_filt = robustified_cbf
        self.t_prop = t_prop
        self.keep_inds = keep_inds

    def __call__(self, obs):
        if self.keep_inds is None:
            v = self.v_filt(obs)
        else:
            v = self.v_filt(obs[:, self.keep_inds])
        # pos err, vel err (from zero), vel des, quat
        rh_obs = torch.cat((v * self.t_prop, obs[..., 7:9], v, obs[..., 3:7]), dim=-1)
        return self.rh(rh_obs)

class DoubleSingleVelTracking:
    def __init__(self, k_d, robustified_cbf):
        self.k_d = k_d
        self.v_filt = robustified_cbf
        self.keep_inds = None

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
    

class RaibertHeuristic:
    def __init__(self, K_p, K_v, K_ff, clip_pos, clip_vel, clip_vel_des, clip_ang, randomize_rp=-0., num_robots=1):
        self.K_p = K_p
        self.K_v = K_v
        self.K_ff = K_ff
        self.clip_pos = clip_pos
        self.clip_vel = clip_vel
        self.clip_vel_des = clip_vel_des
        self.clip_ang = clip_ang
        self.randomize_rp = randomize_rp > 0
        self.num_robots = None
        self.rnd_pitch = torch.rand((num_robots, 1), device="cuda:0") * randomize_rp
        self.rnd_roll = torch.rand((num_robots, 1), device="cuda:0") * randomize_rp


    def get_inference_policy(self, device):
        def policy(obs):
            return RaibertHeuristic.raibert_policy(
                obs, self.K_p, self.K_v, self.K_ff, self.clip_pos, self.clip_vel, self.clip_vel_des, self.clip_ang, 0, 0
            )

        def rnd_policy(obs):
            return RaibertHeuristic.raibert_policy(
                obs, self.K_p, self.K_v, self.K_ff, self.clip_pos, self.clip_vel, self.clip_vel_des, self.clip_ang, self.rnd_pitch, self.rnd_roll
            )

        if self.randomize_rp:
            return policy
        else:
            return rnd_policy

    @staticmethod
    def omega_to_quat(omega_pitch, omega_roll, omega_yaw):
        cy = torch.cos(omega_yaw * 0.5)
        sy = torch.sin(omega_yaw * 0.5)
        cp = torch.cos(omega_pitch * 0.5)
        sp = torch.sin(omega_pitch * 0.5)
        cr = torch.cos(omega_roll * 0.5)
        sr = torch.sin(omega_roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack((w, x, y, z), dim=-1)

    @staticmethod
    def raibert_policy(obs, Kp, Kv, K_ff, clip_pos, clip_vel, clip_vel_des, clip_ang, rnd_pitch, rnd_roll):
        e_x = torch.clamp(obs[:, 0], -clip_pos, clip_pos)
        e_y = torch.clamp(-obs[:, 1], -clip_pos, clip_pos)
        ev_x = torch.clamp(-obs[:, 2], -clip_vel, clip_vel)
        ev_y = torch.clamp(obs[:, 3], -clip_vel, clip_vel)
        vd_x = torch.clamp(obs[:, 4], -clip_vel_des, clip_vel_des)
        vd_y = torch.clamp(-obs[:, 5], -clip_vel_des, clip_vel_des)

        pitch = -Kp * e_x - Kv * ev_x + K_ff * vd_x + rnd_pitch
        roll = -Kp * e_y - Kv * ev_y + K_ff * vd_y + rnd_roll

        current_yaw = RaibertHeuristic.quat_to_yaw(obs[:, 6:10]) * 0.

        omega_quat = RaibertHeuristic.omega_to_quat(
            torch.clamp(pitch, -clip_ang, clip_ang),
            torch.clamp(roll, -clip_ang, clip_ang),
            current_yaw
        )

        return omega_quat

    @staticmethod
    def quat_to_yaw(quat):
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw
