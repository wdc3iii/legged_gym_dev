import torch


class RaibertHeuristic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.K_p = self.cfg.controller.K_p
        self.K_v = self.cfg.controller.K_v
        self.K_ff = self.cfg.controller.K_ff
        self.clip_pos = self.cfg.controller.clip_pos
        self.clip_vel = self.cfg.controller.clip_vel
        self.clip_vel_des = self.cfg.controller.clip_vel_des
        self.clip_ang = self.cfg.controller.clip_ang

    def get_inference_policy(self, device):
        def policy(obs):
            return RaibertHeuristic.raibert_policy(
                obs, self.K_p, self.K_v, self.K_ff, self.clip_pos, self.clip_vel, self.clip_vel_des, self.clip_ang
            )

        return policy

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
    def raibert_policy(obs, Kp, Kv, K_ff, clip_pos, clip_vel, clip_vel_des, clip_ang):
        e_x = torch.clamp(obs[:, 0], -clip_pos, clip_pos)
        e_y = torch.clamp(-obs[:, 1], -clip_pos, clip_pos)
        ev_x = torch.clamp(-obs[:, 2], -clip_vel, clip_vel)
        ev_y = torch.clamp(obs[:, 3], -clip_vel, clip_vel)
        vd_x = torch.clamp(obs[:, 4], -clip_vel_des, clip_vel_des)
        vd_y = torch.clamp(-obs[:, 5], -clip_vel_des, clip_vel_des)

        pitch = -Kp * e_x - Kv * ev_x + K_ff * vd_x
        roll = -Kp * e_y - Kv * ev_y + K_ff * vd_y

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


class DoubleSingleTracking:

    def __init__(self, Kp, Kd, state_dependent_input_bound):
        self.K_p = Kp
        self.K_d = Kd
        self.state_dependent_input_bound = state_dependent_input_bound

    def __call__(self, obs):
        xt = obs[:, :4]
        zt = obs[:, 4:6]
        vt = obs[:, 6:]
        u = self.K_p * (zt - xt[:, :2]) + self.K_d * (vt - xt[:, 2:])
        return self.state_dependent_input_bound(xt, u)
