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

from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch
from typing import Dict
from legged_gym.envs import LeggedRobotTrajectory
from legged_gym.envs.hopper.flat_trajectory.hopper_trajectory_config import HopperRoughTrajectoryCfg
from legged_gym.utils.helpers import torch_rand_vec_float
from pytorch3d.transforms import quaternion_invert, quaternion_multiply, so3_log_map, quaternion_to_matrix, Rotate, \
    euler_angles_to_matrix, matrix_to_quaternion


class HopperTrajectory(LeggedRobotTrajectory):

    def __init__(self, cfg: HopperRoughTrajectoryCfg, sim_params, physics_engine, sim_device, headless):
        self.nominal_spring_stiffness = cfg.asset.spring_stiffness
        self.nominal_spring_damping = cfg.asset.spring_damping
        self.nominal_spring_setpoint = cfg.control.foot_pos_des
        self.stiffness_range = cfg.domain_rand.spring_properties.stiffness_range
        self.damping_range = cfg.domain_rand.spring_properties.damping_range
        self.setpoint_range = cfg.domain_rand.spring_properties.setpoint_range
        self.p_gain_range = cfg.domain_rand.pd_gain_properties.p_gain_range
        self.d_gain_range = cfg.domain_rand.pd_gain_properties.d_gain_range
        self.max_torque_range = cfg.domain_rand.torque_speed_properties.max_torque_range
        self.max_speed_range = cfg.domain_rand.torque_speed_properties.max_speed_range
        self.max_slope_range = cfg.domain_rand.torque_speed_properties.slope_range
        self.zero_action = torch.repeat_interleave(torch.tensor(cfg.control.zero_action).reshape((1, -1)), cfg.env.num_envs, 0).float()
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.zero_action = self.zero_action.to(self.device)
        self.foot_joint_index = torch.tensor([0])
        mask = torch.ones(self.num_dof, dtype=torch.bool)
        mask[self.foot_joint_index] = False
        self.wxyz_quat_inds = torch.tensor([6, 3, 4, 5])
        self.wheel_joint_indices = torch.arange(self.num_dof)[mask]

        self.spring_stiffness = torch.ones((self.num_envs, 1), device=self.device) * self.nominal_spring_stiffness
        self.spring_damping = torch.ones((self.num_envs, 1), device=self.device) * self.nominal_spring_damping
        self.actuator_transform = Rotate(torch.tensor(self.cfg.asset.rot_actuator), device=self.device)
        self.torque_speed_bound_ratio = self.cfg.asset.torque_speed_bound_ratio
        self.foot_pos_des = torch.ones((self.num_envs, 1), device=self.device) * self.nominal_spring_setpoint

        self.default_dof_pos_noise_lower = torch.tensor(self.cfg.init_state.default_dof_pos_noise_lower,
                                                        device=self.device)
        self.default_dof_pos_noise_upper = torch.tensor(self.cfg.init_state.default_dof_pos_noise_upper,
                                                        device=self.device)
        self.default_dof_vel_noise_lower = torch.tensor(self.cfg.init_state.default_dof_vel_noise_lower,
                                                        device=self.device)
        self.default_dof_vel_noise_upper = torch.tensor(self.cfg.init_state.default_dof_vel_noise_upper,
                                                        device=self.device)
        self.default_root_pos_noise_lower = torch.tensor(self.cfg.init_state.default_root_pos_noise_lower,
                                                         device=self.device)
        self.default_root_pos_noise_upper = torch.tensor(self.cfg.init_state.default_root_pos_noise_upper,
                                                         device=self.device)
        self.default_root_vel_noise_lower = torch.tensor(self.cfg.init_state.default_root_vel_noise_lower,
                                                         device=self.device)
        self.default_root_vel_noise_upper = torch.tensor(self.cfg.init_state.default_root_vel_noise_upper,
                                                         device=self.device)
        self.max_vel = torch.tensor(self.cfg.domain_rand.max_push_vel, device=self.device)
        self._update_envs()

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

            # Hopper-specific
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

        self.time_until_next_push -= self.cfg.control.decimation * self.sim_params.dt
        need_push = self.time_until_next_push <= 0

        if self.cfg.domain_rand.push_robots and torch.any(need_push):
            self._push_robots(need_push.nonzero(as_tuple=False).flatten())
            # Reset the timer for the next push for the environments that needed a push
            lb = self.cfg.domain_rand.time_between_pushes[0]
            ub = self.cfg.domain_rand.time_between_pushes[1]
            if self.cfg.curriculum.use_curriculum:
                lb *= self.cfg.curriculum.push.time[self.curriculum_state]
                ub *= self.cfg.curriculum.push.time[self.curriculum_state]
            self.time_until_next_push[need_push] = torch_rand_float(lb,
                                                                    ub,
                                                                    (torch.sum(need_push), 1),
                                                                    device=self.device).flatten()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.prev_error[:] = torch.square(self.trajectory[:, 0, :] - self.rom.proj_z(self.root_states))

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

        p_gains = self.p_gains * self.p_gain_random
        d_gains = self.d_gains * self.d_gain_random
        kd_spindown = self.kd_spindown * self.d_gain_random[:, self.wheel_joint_indices]
        # Compute the foot control action unless the RL agent is responsible for this
        if "w_foot" in control_type:
            self.torques[:, self.foot_joint_index] = -p_gains[:, self.foot_joint_index] * (foot_pos - actions_scaled[:, -1]) - d_gains[:, self.foot_joint_index] * foot_vel
        else:
            self.torques[contact_inds, self.foot_joint_index] = 0
            self.torques[not_contact_inds, self.foot_joint_index] = -p_gains[not_contact_inds, self.foot_joint_index] * (foot_pos[not_contact_inds.squeeze()] - self.foot_pos_des[not_contact_inds.squeeze()]) - d_gains[not_contact_inds, self.foot_joint_index] * foot_vel[not_contact_inds.squeeze()]
        # Add in foot spring force
        self.torques[contact_inds, self.foot_joint_index] += -self.spring_stiffness[contact_inds.squeeze()] * foot_pos[contact_inds.squeeze()] - self.spring_damping[contact_inds.squeeze()] * foot_vel[contact_inds.squeeze()]

        # Compute wheel torques
        if "spindown" in control_type:
            self.torques[contact_inds, self.wheel_joint_indices] = -kd_spindown[contact_inds.squeeze(), :] * wheel_vel[contact_inds.squeeze()]
            orient_inds = not_contact_inds.squeeze()
        else:
            orient_inds = torch.arange(self.num_envs)

        if orient_inds.shape != torch.Size([]):
            if orient_inds.shape[0] > 0:
                if "orientation" in control_type:
                    quat_des = actions_scaled[orient_inds, :] / torch.linalg.norm(actions_scaled[orient_inds, :], dim=-1, keepdim=True)

                    quat_act = self.root_states[orient_inds[:, None], self.wxyz_quat_inds]
                    err = quaternion_multiply(quaternion_invert(quat_des), quat_act)
                    log_err = so3_log_map(quaternion_to_matrix(err))
                    local_tau = -p_gains[orient_inds[:, None], self.wheel_joint_indices] * log_err - d_gains[orient_inds[:, None], self.wheel_joint_indices] * self.base_ang_vel[orient_inds.squeeze(), :]

                    tau = self.actuator_transform.transform_points(local_tau)
                    self.torques[orient_inds[:, None], self.wheel_joint_indices] = tau
                elif "V" in control_type:
                    self.torques[orient_inds, self.wheel_joint_indices] = -p_gains[orient_inds[:, None], self.wheel_joint_indices] * (actions_scaled[orient_inds, self.wheel_joint_indices] - wheel_vel) \
                                                          - d_gains[orient_inds[:, None], self.wheel_joint_indices] * (wheel_vel - self.last_dof_vel[orient_inds, self.wheel_joint_indices]) / self.sim_params.dt
                elif "T" in control_type:
                    self.torques[orient_inds, self.wheel_joint_indices] = actions_scaled[orient_inds, self.wheel_joint_indices]
                else:
                    raise NameError(f"Unknown controller type: {control_type}")

        ts_ratio = self.torque_speed_bound_ratio * self.torque_speed_bound_ratio_random
        t_bound = self.torque_limits * self.torque_limit_random
        w_bound = self.wheel_speed_limits * self.wheel_limit_random
        state_input_upper = -ts_ratio * t_bound[:, self.wheel_joint_indices] / w_bound * (wheel_vel - w_bound)
        state_input_lower = -ts_ratio * t_bound[:, self.wheel_joint_indices] / w_bound * (wheel_vel + w_bound)
        self.torques[:, self.wheel_joint_indices] = torch.clip(self.torques[:, self.wheel_joint_indices],
                                                               state_input_lower, state_input_upper)
        return torch.clip(self.torques, -t_bound, t_bound)

    def compute_observations(self):
        """ Computes observations
        """
        actions_normalized_quat = torch.clone(self.actions)
        actions_normalized_quat /= torch.linalg.norm(actions_normalized_quat, dim=-1,
                                                     keepdim=True)  # Normalize actions when feeding back to model
        actions_normalized_quat[actions_normalized_quat[:, 0] < 0, :] *= -1  # By convention, always have qw > 0
        # Adjust trajectory positions relative to current position
        mod_traj = torch.clone(self.trajectory)
        mod_traj[:, :, :2] -= self.rom.proj_z(self.root_states)[:, None, :2]
        #
        # if self.base_quat[3] > 0:
        #     print('pos')

        self.obs_buf = torch.cat((self.root_states[:, 2][:, None] * self.obs_scales.z_pos,
                                  self.base_quat,
                                  self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.dof_vel[:, self.wheel_joint_indices] * self.obs_scales.dof_vel,
                                  (mod_traj * self.trajectory_scale).reshape(self.num_envs, -1),
                                  actions_normalized_quat
                                  ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(self.zero_action)
        # For some reason we need an addiontional reset
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(self.zero_action)
        return obs, privileged_obs

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_vec_float(
            self.default_dof_pos_noise_lower, self.default_dof_pos_noise_upper,
            (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = torch_rand_vec_float(
            self.default_dof_vel_noise_lower, self.default_dof_vel_noise_upper,
            (len(env_ids), self.num_dof), device=self.device
        )
        self.actions[env_ids, :] = torch.clone(self.zero_action[env_ids, :].detach())

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :7] += torch_rand_vec_float(
                self.default_root_pos_noise_lower, self.default_root_pos_noise_upper,
                (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
            self.root_states[env_ids, 3:7] /= torch.linalg.norm(self.root_states[env_ids, 3:7], dim=-1, keepdim=True)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 2:7] += torch_rand_vec_float(
                self.default_root_pos_noise_lower[2:], self.default_root_pos_noise_upper[2:],
                (len(env_ids), 5), device=self.device
            )  # xy position within 1m of the center
            self.root_states[env_ids, 3:7] /= torch.linalg.norm(self.root_states[env_ids, 3:7], dim=-1, keepdim=True)
        if self.cfg.init_state.randomize_yaw:
            yaw = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
            quat_yaw = matrix_to_quaternion(euler_angles_to_matrix(torch.concatenate([torch.zeros((len(env_ids), 2), device=self.device), yaw], dim=-1), "XYZ"))
            q_wxyz = self.root_states[:, self.wxyz_quat_inds][env_ids, :]
            q_new_wxyz = quaternion_multiply(q_wxyz, quat_yaw)
            self.root_states[env_ids, 3:7] = q_new_wxyz[:, [1, 2, 3, 0]]

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_vec_float(
            self.default_root_vel_noise_lower, self.default_root_vel_noise_upper,
            (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self, push_idx):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """

        self.root_states[push_idx, 7:13] = torch_rand_vec_float(-self.max_vel, self.max_vel,
                                                                (len(push_idx), 6), device=self.device)

        env_ids_int32 = push_idx.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                             gymtorch.unwrap_tensor(self.root_states),
                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_envs(self):
        super()._update_envs()
        self._process_spring_properties()
        self._process_torque_speed_properties()
        self._process_pd_gain_properties()

    def _process_spring_properties(self):
        if self.cfg.domain_rand.spring_properties.randomize_stiffness:
            self.spring_stiffness = torch_rand_float(self.stiffness_range[0], self.stiffness_range[1],
                                                     (self.num_envs, 1), self.device) * self.nominal_spring_stiffness
        if self.cfg.domain_rand.spring_properties.randomize_damping:
            self.spring_damping = torch_rand_float(self.damping_range[0], self.damping_range[1],
                                                     (self.num_envs, 1), self.device) * self.nominal_spring_damping
        if self.cfg.domain_rand.spring_properties.randomize_setpoint:
            self.foot_pos_des = torch_rand_float(self.setpoint_range[0], self.setpoint_range[1],
                                                     (self.num_envs, 1), self.device) * self.nominal_spring_setpoint

    def _process_pd_gain_properties(self):
        if self.cfg.domain_rand.pd_gain_properties.randomize_p_gain:
            self.p_gain_random = torch_rand_float(self.p_gain_range[0], self.p_gain_range[1], (self.num_envs, 4), device=self.device)
        else:
            self.p_gain_random = torch.ones((self.num_envs, 4), device=self.device)
        if self.cfg.domain_rand.pd_gain_properties.randomize_d_gain:
            self.d_gain_random = torch_rand_float(self.d_gain_range[0], self.d_gain_range[1], (self.num_envs, 4), device=self.device)
        else:
            self.d_gain_random = torch.ones((self.num_envs, 4), device=self.device)

    def _process_torque_speed_properties(self):
        if self.cfg.domain_rand.torque_speed_properties.randomize_slope:
            self.torque_speed_bound_ratio_random = torch_rand_float(self.max_slope_range[0], self.max_slope_range[1], (self.num_envs, 1), device=self.device)
        else:
            self.torque_speed_bound_ratio_random = torch.ones((self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.torque_speed_properties.randomize_max_torque:
            self.torque_limit_random = torch_rand_float(self.max_torque_range[0], self.max_torque_range[1], (self.num_envs, 4), device=self.device)
        else:
            self.torque_limit_random = torch.ones((self.num_envs, 4), device=self.device)
        if self.cfg.domain_rand.torque_speed_properties.randomize_max_speed:
            self.wheel_limit_random = torch_rand_float(self.max_speed_range[0], self.max_speed_range[1], (self.num_envs, 3), device=self.device)
        else:
            self.wheel_limit_random = torch.ones((self.num_envs, 3), device=self.device)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed_old quantities
        """
        super()._init_buffers()
        self.actions = torch.clone(self.zero_action.to(self.device).detach())
        self.kd_spindown = torch.zeros(3, dtype=torch.float, device=self.device, requires_grad=False)
        self.wheel_speed_limits = torch.zeros(3, dtype=torch.float, device=self.device, requires_grad=False)
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

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0] = noise_scales.z_pos * noise_level * self.obs_scales.z_pos
        noise_vec[1:5] = noise_scales.quat * noise_level
        noise_vec[5:8] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[8:11] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[11:14] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        traj_size = self.traj_gen.N * self.rom.n
        noise_vec[14:14 + traj_size] = 0.  # commands

        noise_vec[14 + traj_size:26 + traj_size] = 0.  # previous actions

        if self.cfg.terrain.measure_heights:
            noise_vec[26 + traj_size:213 + traj_size] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        return noise_vec

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(torch.abs(self.torques[:, self.wheel_joint_indices]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel[:, self.wheel_joint_indices] - self.dof_vel[:, self.wheel_joint_indices]) / self.dt), dim=1)

    def _reward_unit_quat(self):
        act_norm = torch.linalg.norm(self.actions, dim=-1)
        return torch.square(1 - act_norm)
