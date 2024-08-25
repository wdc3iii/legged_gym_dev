import torch


class RaibertHeuristic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.K_p = self.cfg.controller.K_p
        self.K_v = self.cfg.controller.K_v
        self.K_ff = self.cfg.controller.K_ff
        self.clip_value_pos = self.cfg.controller.clip_value_pos
        self.clip_value_vel = self.cfg.controller.clip_value_vel
        self.clip_value_total = self.cfg.controller.clip_value_total

    def get_inference_policy(self, device):
        def policy(obs):
            return RaibertHeuristic.raibert_policy(
                obs, self.K_p, self.K_v, self.K_ff, self.clip_value_pos, self.clip_value_vel, self.clip_value_total
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
    def raibert_policy(obs, Kp, Kv, K_ff, clip_pos, clip_vel, clip_ang):
        pos_error_x = obs[:, 0]
        pos_error_y = -obs[:, 1]
        cur_err_vel_x = -obs[:, 2]
        cur_err_vel_y = obs[:, 3]
        des_vel_x = obs[:, 4]
        des_vel_y = -obs[:, 5]

        pitch_pos_scaled = -Kp * pos_error_x
        roll_pos_scaled = -Kp * pos_error_y
        vel_x_scaled = -Kv * cur_err_vel_x + K_ff * des_vel_x
        vel_y_scaled = -Kv * cur_err_vel_y + K_ff * des_vel_y

        pitch_pos = torch.clamp(pitch_pos_scaled, -clip_pos, clip_pos)
        roll_pos = torch.clamp(roll_pos_scaled, -clip_pos, clip_pos)
        vel_x_clamped = torch.clamp(vel_x_scaled, -clip_vel, clip_vel)
        vel_y_clamped = torch.clamp(vel_y_scaled, -clip_vel, clip_vel)
        omega_pitch = pitch_pos + vel_x_clamped
        omega_roll = roll_pos + vel_y_clamped

        omega_pitch = torch.clamp(omega_pitch, -clip_ang, clip_ang)
        omega_roll = torch.clamp(omega_roll, -clip_ang, clip_ang)

        current_yaw = RaibertHeuristic.quat_to_yaw(obs[:, 6:10])

        omega_quat = RaibertHeuristic.omega_to_quat(omega_pitch, omega_roll, current_yaw)

        return omega_quat

    @staticmethod
    def quat_to_yaw(quat):
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw
