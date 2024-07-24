# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.envs.hopper.hopper_config import HopperRoughCfg
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, so3_log_map, quaternion_to_matrix, Rotate


class Hopper(LeggedRobot):

    def __init__(self, cfg: HopperRoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.foot_joint_index = torch.tensor([0])
        mask = torch.ones(self.num_dof, dtype=torch.bool)
        mask[self.foot_joint_index] = False
        self.wxyz_quat_inds = torch.tensor([6, 3, 4, 5])
        self.wheel_joint_indices = torch.arange(self.num_dof)[mask]
        self.spring_stiffness = self.cfg.asset.spring_stiffness
        self.spring_damping = self.cfg.asset.spring_damping
        self.actuator_transform = Rotate(torch.tensor(self.cfg.asset.rot_actuator), device=self.device)
        self.torque_speed_bound_ratio = self.cfg.asset.torque_speed_bound_ratio
        self.foot_pos_des = self.cfg.control.foot_pos_des
        self.zero_action = torch.repeat_interleave(torch.tensor(cfg.control.zero_action).reshape((1, -1)), self.num_envs, 0)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # prepare quantities
            self.base_quat[:] = self.root_states[:, 3:7]
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        control_type = self.cfg.control.control_type
        actions_scaled = actions * self.cfg.control.action_scale

        foot_pos = self.dof_pos[:, self.foot_joint_index]
        foot_vel = self.dof_vel[:, self.foot_joint_index]
        contacts = torch.squeeze(self.contact_forces[:, self.feet_indices, 2] > 0.1)

        not_contacts = torch.logical_not(contacts)
        contact_inds = torch.nonzero(contacts, as_tuple=False)
        not_contact_inds = torch.nonzero(not_contacts, as_tuple=False)
        wheel_vel = self.dof_vel[:, self.wheel_joint_indices]

        # Compute the foot control action unless the RL agent is responsible for this
        if "w_foot" in control_type:
            self.torques[:, self.foot_joint_index] = -self.p_gains[self.foot_joint_index] * (foot_pos - actions_scaled[:, -1]) - self.d_gains[self.foot_joint_index] * foot_vel
        else:
            self.torques[contact_inds, self.foot_joint_index] = 0
            self.torques[not_contact_inds, self.foot_joint_index] = -self.p_gains[self.foot_joint_index] * (foot_pos[not_contact_inds.squeeze()] - self.foot_pos_des) - self.d_gains[self.foot_joint_index] * foot_vel[not_contact_inds.squeeze()]
        # Add in foot spring force
        self.torques[contact_inds, self.foot_joint_index] += -self.spring_stiffness * foot_pos[contact_inds.squeeze()] - self.spring_damping * foot_vel[contact_inds.squeeze()]

        # Compute wheel torques
        if "spindown" in control_type:
            self.torques[contact_inds, self.wheel_joint_indices] = -self.kd_spindown * wheel_vel[contact_inds.squeeze()]
            orient_inds = not_contact_inds.squeeze()
        else:
            orient_inds = torch.arange(self.num_envs)

        if "orientation" in control_type:
            quat_des = actions_scaled[orient_inds, :] / torch.linalg.norm(actions_scaled[orient_inds, :], dim=-1, keepdim=True)
            # quat_des = torch.zeros_like(actions_scaled[orient_inds, :]).to(self.device)
            # quat_des[:, 0] = 1

            quat_act = self.root_states[orient_inds[:, None], self.wxyz_quat_inds]
            err = quaternion_multiply(quaternion_invert(quat_des), quat_act)
            log_err = so3_log_map(quaternion_to_matrix(err))
            local_tau = -self.p_gains[self.wheel_joint_indices] * log_err - self.d_gains[self.wheel_joint_indices] * self.base_ang_vel[orient_inds.squeeze(), :]

            # local_tau[:, :] = torch.tensor([-0.8165, 0, -0.5773], device=self.device)
            tau = self.actuator_transform.transform_points(local_tau)
            self.torques[orient_inds[:, None], self.wheel_joint_indices] = tau
        elif "V" in control_type:
            self.torques[orient_inds, self.wheel_joint_indices] = -self.p_gains[self.wheel_joint_indices] * (actions_scaled[orient_inds, self.wheel_joint_indices] - wheel_vel) \
                                                  - self.d_gains[self.wheel_joint_indices] * (wheel_vel - self.last_dof_vel[orient_inds, self.wheel_joint_indices]) / self.sim_params.dt
        elif "T" in control_type:
            self.torques[orient_inds, self.wheel_joint_indices] = actions_scaled[orient_inds, self.wheel_joint_indices]
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        state_input_upper = -self.torque_speed_bound_ratio * self.torque_limits[self.wheel_joint_indices] / self.wheel_speed_limits * (wheel_vel - self.wheel_speed_limits)
        state_input_lower = -self.torque_speed_bound_ratio * self.torque_limits[self.wheel_joint_indices] / self.wheel_speed_limits * (wheel_vel + self.wheel_speed_limits)
        self.torques[:, self.wheel_joint_indices] = torch.clip(self.torques[:, self.wheel_joint_indices], state_input_lower, state_input_upper)
        return torch.clip(self.torques, -self.torque_limits, self.torque_limits)

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(self.zero_action)
        return obs, privileged_obs

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()
        self.kd_spindown = torch.zeros(3, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_speed_limits = torch.zeros(3, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(1, 4):
            wheel_str = f"wheel{i}_rotation"
            if  wheel_str in self.cfg.control.wheel_spindown.keys():
                self.kd_spindown[i - 1] = self.cfg.control.wheel_spindown[wheel_str]
            else:
                print(f"Spin down gain of joint {wheel_str} were not defined, setting them to zero")
            if  wheel_str in self.cfg.asset.wheel_speed_bounds.keys():
                self.wheel_speed_limits[i - 1] = self.cfg.asset.wheel_speed_bounds[wheel_str]
            else:
                print(f"wheel speed bound {wheel_str} were not defined, setting them to infinite")
                self.wheel_speed_limits[i - 1] = torch.inf

        self.torques = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32).to(self.device)

    # def _reward_upright_orientation(self):
    #     """Penalty for deviation from upright orientation, returning a negative exponential value."""
    #     upright_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
    #     cos_theta = torch.abs(torch.sum(self.base_quat * upright_quat, dim=1))
    #     angle_rad = torch.acos(cos_theta) * 2  # Convert from half-angle to full angle in radians
    #     penalty = torch.exp(angle_rad)  # Exponential penalty
    #     return torch.exp(-upright_error / self.cfg.rewards.upright_orientation_sigma)
