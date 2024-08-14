import torch
from scipy.spatial.transform import Rotation as R


class RaibertHeuristic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.K_p = -.1  # Proportional gain for position
        self.K_v = -.3  # Proportional gain for velocity
        self.clip_value_pos = 1.  # Clipping value for position errors
        self.clip_value_vel = 1.  # Clipping value for velocity errors
        self.clip_value_total = 1.  # Clipping value for total combination of errors

    def get_inference_policy(self, device):
        def policy(obs):
            pos_error_x = obs[:, 0]
            pos_error_y = obs[:, 1]
            vel_error_x = obs[:, 2]
            vel_error_y = obs[:, 3]

            pitch_pos_scaled = -self.K_p * pos_error_x
            roll_pos_scaled = -self.K_p * pos_error_y
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

            current_yaw = self.quat_to_yaw(obs[:, 4:8])

            omega_quat = self.omega_to_quat(omega_pitch, omega_roll, current_yaw)

            return omega_quat

        return policy

    def omega_to_quat(self, omega_pitch, omega_roll, omega_yaw):
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
                                                                            # TODO: Figure this out
        return torch.stack((w, x, y, z), dim=-1)                    # when in this config, wich quat_to_yaw being in x,y,z,w order, it doesnt spin but torque curves are wrong
        # return torch.stack((x, y, z, w), dim=-1)                          # when in this config, torque curves are correct, but it spins out and almost turns upside down

    @staticmethod
    def quat_to_yaw(quat):
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return yaw
