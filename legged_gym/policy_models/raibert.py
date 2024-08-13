import torch
from scipy.spatial.transform import Rotation as R


class RaibertHeuristic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.K_p = .1 # cfg.policy_model.rh.K_p  # Proportional gain for position
        self.K_v = .3 # cfg.policy_model.rh.K_v  # Proportional gain for velocity
        self.clip_value_pos = 1.  # cfg.policy_model.rh.clip_value_pos  # Clipping value for position errors
        self.clip_value_vel = 1.  # cfg.policy_model.rh.clip_value_vel  # Clipping value for velocity errors
        self.clip_value_total = 1.  # cfg.policy_model.rh.clip_value_total  # Clipping value for total combination of errors

    def get_inference_policy(self, device):
        def policy(obs):
            base_lin_vel = obs[:, :3]

            traj_pos_x = obs[:, 9]
            traj_pos_y = obs[:, 10]
            traj_vel_x = obs[:, 11]
            traj_vel_y = obs[:, 12]
            vel_error_x = traj_vel_x - base_lin_vel[:, 0]
            vel_error_y = traj_vel_y - base_lin_vel[:, 1]

            pitch_pos_scaled = -self.K_p * traj_pos_x
            roll_pos_scaled = -self.K_p * traj_pos_y
            vel_x_scaled = -self.K_v * vel_error_x
            vel_y_scaled = -self.K_v * vel_error_y

            pitch_pos = torch.clamp(pitch_pos_scaled, -self.clip_value_pos, self.clip_value_pos)
            roll_pos = torch.clamp(roll_pos_scaled, -self.clip_value_pos, self.clip_value_pos)
            vel_x_clamped = torch.clamp(vel_x_scaled, -self.clip_value_vel, self.clip_value_vel)
            vel_y_clamped = torch.clamp(vel_y_scaled, -self.clip_value_vel, self.clip_value_vel)
            omega_pitch = pitch_pos + vel_x_clamped
            omega_roll = roll_pos + vel_y_clamped

            omega_pitch = torch.clamp(omega_pitch, -self.clip_value_total, self.clip_value_total)
            omega_roll = torch.clamp(omega_roll, -self.clip_value_total, self.clip_value_total)

            omega_quat = self.omega_to_quat(omega_pitch, omega_roll, device)

            return omega_quat

        return policy

    def omega_to_quat(self, omega_pitch, omega_roll, device):
        cy = torch.ones_like(omega_roll)
        sy = torch.zeros_like(omega_roll)
        cp = torch.cos(omega_pitch * 0.5)
        sp = torch.sin(omega_pitch * 0.5)
        cr = torch.cos(omega_roll * 0.5)
        sr = torch.sin(omega_roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack((x, y, z, w), dim=-1)

