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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os


from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
import math

from isaacgym.torch_utils import *
class LeggedRobot(BaseTask):
    def __init__(
            self,
            cfg: LeggedRobotCfg,
            sim_params,
            physics_engine,
            sim_device,
            headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self.group_idx = torch.arange(0, self.cfg.env.num_envs)
        self._prepare_reward_function()
        self.init_done = True

    def get_observations_history(self):
        ''' Overide the base class method to return the observations and observation history
        '''
        return self.obs_history_buf

    def reset(self):
        ''' Reset all robots (overide the base class method)
        '''
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs,

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
            if self.cfg.domain_rand.randomize_action_delay:
                self.action_fifo = torch.cat((self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1)
                self.torques = self._compute_torques(self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
                                                    ).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.obs_history_buf

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]

        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.power = torch.abs(self.torques * self.dof_vel)
        self._post_physics_step_callback()
        self.foot_heights = torch.clip((self.foot_positions[:, :, 2]- 0.022- self._get_foot_heights()),0,1,)
        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_feet_states()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # in some cases a simulation step might be required to refresh some obs
        # (for example body positions)
        self.compute_observations()
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_position[:] = self.base_position[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # def check_termination(self):
    #     """ Check if environments need to be reset
    #     """
    #     self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    #     self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
    #     long_time_trap = self.trap_static_time > 5 # seconds
    #     self.large_ori_buf = self.projected_gravity[:, 2] > 0

    #     self.reset_buf |= long_time_trap
    #     self.reset_buf |= self.large_ori_buf
    #     self.reset_buf |= self.time_out_buf
    def check_termination(self):
        """Check if environments need to be reset"""
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        fail_buf |= self.projected_gravity[:, 2] > -0.1
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.power_limit_out_buf = (
            torch.sum(self.power, dim=1) > self.cfg.control.max_power
        )
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.edge_reset_buf = self.base_position[:, 0] > self.terrain_x_max - 1
            self.edge_reset_buf |= self.base_position[:, 0] < self.terrain_x_min + 1
            self.edge_reset_buf |= self.base_position[:, 1] > self.terrain_y_max - 1
            self.edge_reset_buf |= self.base_position[:, 1] < self.terrain_y_min + 1
        long_time_trap = self.trap_static_time > 5 
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
            | self.edge_reset_buf
            | long_time_trap
            # | self.power_limit_out_buf
        )

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum
        # command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids,self.cfg)

        # reset buffers
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.obs_history_buf[env_ids, :] = 0. # reset obs history
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.trap_static_time[env_ids] = 0.
        self.fail_buf[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.terrain.curriculum:
            # self.extras["episode"]["group_terrain_level"] = torch.mean(
            #     self.terrain_levels[self.group_idx].float()
            # )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
            self.extras["episode"]["m_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.none_smooth_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_feet_states(self):
        ''' compute the relative feet positions and velocities
        '''
        self.foot_positions = self.rigid_body_state[:, self.feet_indices, :3] 
        self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10] 

    def compute_proprioceptive_observations(self):
        """ Computes privileged observations
        """
        self.proprioceptive_obs_buf = torch.cat((  
                                                self.base_ang_vel  * self.obs_scales.ang_vel,
                                                self.projected_gravity,
                                                self.commands[:, :3] * self.commands_scale,
                                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                                self.dof_vel * self.obs_scales.dof_vel,
                                                self.actions
                                                ),dim=-1)
        return self.proprioceptive_obs_buf
    
    def compute_observations(self):
        """ Computes observations
        """
        rigid_body_states = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.obs_buf = self.compute_proprioceptive_observations()
        # add noise to proprioceptive observations
         
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        #ang_vel:[0.06]
        if self.cfg.env.num_privileged_obs is not None:
            self.adapt_observations = torch.cat((
                                        self.Kp_factors,#12
                                        self.Kd_factors,#12
                                        self.friction_coeffs_tensor,#1
                                        self.restitution_tensor,#1

                                        # self.leg_params_tensor,#4
                                        # self.mass_params_tensor,#10
                                        # self.motor_strength[0] - 1, #12
                                        self.motor_strengths #12
                                        ),dim=-1)
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.25 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements

            self.privileged_obs_buf = torch.cat((
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    # self.base_ang_vel  * self.obs_scales.ang_vel,
                    self.obs_buf,
                    heights,
                    self.adapt_observations,
                    self.torques,
                    (self.last_dof_vel - self.dof_vel) / self.dt,
                    self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1)
                    ),dim=-1)

        # left (far recent) to right (close recent)
        self.obs_history_buf = torch.cat((self.obs_history_buf[:, self.obs_buf.shape[1]:], self.obs_buf), dim=1) 
        if self.cfg.domain_rand.randomize_imu_offset:
            randomized_base_quat = quat_mul(self.random_imu_offset, self.base_quat)
            self.obs_buf[:, :3] = quat_rotate_inverse(randomized_base_quat, self.root_states[:, 10:13]) * self.obs_scales.ang_vel
            self.obs_buf[:, 3:6] = quat_rotate_inverse(randomized_base_quat, self.gravity_vec) 
        

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range

                restitution_range = self.cfg.domain_rand.restitution_range

                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                
                self.friction_coeffs = friction_buckets[bucket_ids]
                self.restitutions = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitutions[env_id]
                props[s].friction = self.friction_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False)
            self.dof_vel_limits = torch.zeros(
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False)
            self.torque_limits = torch.zeros(
                self.num_dof,
                dtype=torch.float,
                device=self.device,
                requires_grad=False)

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        # randomize leg mass
        if self.cfg.domain_rand.randomize_leg_mass:
            rng_leg = self.cfg.domain_rand.added_leg_mass_range
            factor_leg_mass = self.cfg.domain_rand.factor_leg_mass_range
            rand_leg_mass = np.random.uniform(factor_leg_mass[0], factor_leg_mass[1], size=(1, ))
            for i in range(len(props)):
                mess = np.random.uniform(factor_leg_mass[0], factor_leg_mass[1])
                props[i].mass *= mess
        else:
            rand_leg_mass = np.zeros((1, ))

        # randomize leg com
        if self.cfg.domain_rand.randomize_leg_com:
            rng_leg_com = self.cfg.domain_rand.added_leg_com_range
            rand_leg_com = np.random.uniform(rng_leg_com[0], rng_leg_com[1], size=(3, ))
            factor_leg_mass = self.cfg.domain_rand.factor_leg_mass_range

            for i in range(len(props)):
                props[i].inertia.x.x *= np.random.uniform(factor_leg_mass[0], factor_leg_mass[1])
                props[i].inertia.y.y *= np.random.uniform(factor_leg_mass[0], factor_leg_mass[1])
                props[i].inertia.z.z *= np.random.uniform(factor_leg_mass[0], factor_leg_mass[1])
                props[i].com += gymapi.Vec3(*rand_leg_com)
        else:
            rand_leg_com = np.zeros((3))

        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng[0], rng[1], size=(1, ))
            props[0].mass += np.random.uniform(rng[0], rng[1])

        else:
            rand_mass = np.zeros((1, ))
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        else:
            rand_com = np.zeros(3)
        if self.cfg.domain_rand.randomize_base_inertia:
            rng_inertia_xx = self.cfg.domain_rand.added_inertia_range_xx
            rng_inertia_xy = self.cfg.domain_rand.added_inertia_range_xy
            rng_inertia_xz = self.cfg.domain_rand.added_inertia_range_xz
            rng_inertia_yy = self.cfg.domain_rand.added_inertia_range_yy
            rng_inertia_zz = self.cfg.domain_rand.added_inertia_range_zz
            rand_xx = np.random.uniform(rng_inertia_xx[0], rng_inertia_xx[1], size=(1, ))
            rand_xy = np.random.uniform(rng_inertia_xy[0], rng_inertia_xy[1], size=(1, ))
            rand_xz = np.random.uniform(rng_inertia_xz[0], rng_inertia_xz[1], size=(1, ))
            rand_yy = np.random.uniform(rng_inertia_yy[0], rng_inertia_yy[1], size=(1, ))
            rand_zz = np.random.uniform(rng_inertia_zz[0], rng_inertia_zz[1], size=(1, ))
            rand_inertia = np.concatenate([rand_xx,rand_xy,rand_xz,rand_yy,np.array([0]),rand_zz])
            rand_inertia_matrix = gymapi.Mat33() #Mat33 style
            rand_inertia_matrix.x = gymapi.Vec3(rand_inertia[0],rand_inertia[1],rand_inertia[2])
            rand_inertia_matrix.y = gymapi.Vec3(rand_inertia[1],rand_inertia[3],rand_inertia[4])
            rand_inertia_matrix.z = gymapi.Vec3(rand_inertia[2],rand_inertia[4],rand_inertia[5])
            props[0].inertia.x += rand_inertia_matrix.x
            props[0].inertia.y += rand_inertia_matrix.y
            props[0].inertia.z += rand_inertia_matrix.z
        else:
            rand_inertia = np.zeros(6)
        mass_params = np.concatenate([rand_mass, rand_com,rand_inertia])
        leg_params =  np.concatenate([rand_leg_mass,rand_leg_com])
        return props, mass_params, leg_params
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf %
                   int(self.cfg.commands.resampling_time /
                       self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._randomize_dof_props(env_ids, self.cfg)
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
                self.common_step_counter %
                self.cfg.domain_rand.push_interval == 0):

            self._push_robots()

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     self.commands[env_ids,
    #                   0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
    #                                         self.command_ranges["lin_vel_x"][1],
    #                                         (len(env_ids),
    #                                          1),
    #                                         device=self.device).squeeze(1)
    #     self.commands[env_ids,
    #                   1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
    #                                         self.command_ranges["lin_vel_y"][1],
    #                                         (len(env_ids),
    #                                          1),
    #                                         device=self.device).squeeze(1)
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids,
    #                       3] = torch_rand_float(self.command_ranges["heading"][0],
    #                                             self.command_ranges["heading"][1],
    #                                             (len(env_ids),
    #                                              1),
    #                                             device=self.device).squeeze(1)
    #     else:
    #         self.commands[env_ids,
    #                       2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],
    #                                             self.command_ranges["ang_vel_yaw"][1],
    #                                             (len(env_ids),
    #                                              1),
    #                                             device=self.device).squeeze(1)


    #     # set small commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
    #     self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)
        
    #     self.v_level[env_ids] = torch.clip(1*torch.norm(self.commands[env_ids, :2], dim=-1)+0.5*torch.abs(self.commands[env_ids, 2]), min=1)
    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_norm
        # ).unsqueeze(1)
        zero_command_idx = (
            (
                torch_rand_float(0, 1, (len(env_ids), 1), device=self.device)
                > self.cfg.commands.zero_command_prob
            )
            .squeeze(1)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.commands[zero_command_idx, :3] = 0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_idx], self.forward_vec[zero_command_idx]
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_idx, 3] = heading
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)
        
        self.v_level[env_ids] = torch.clip(1.5*torch.norm(self.commands[env_ids, :2], dim=-1)+0.5*torch.abs(self.commands[env_ids, 2]), min=1)
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:,[0,3,6,9]]*=0.5
        self.joint_pos_target = actions_scaled + self.default_dof_pos
        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = self.p_gains * self.Kp_factors * \
                (self.joint_pos_target  -
                 self.dof_pos + self.motor_offsets) - self.d_gains * self.dof_vel*self.Kd_factors
        elif control_type == "V":
            torques = self.p_gains * self.Kp_factors * (actions_scaled - self.dof_vel) - self.d_gains*self.Kd_factors * (
                self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":

            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        torques = torques* self.motor_strengths
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(
                self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32))


    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # xy position within 1m of the center
            self.root_states[env_ids,
                             :2] += torch_rand_float(-1.,
                                                     1.,
                                                     (len(env_ids),
                                                      2),
                                                     device=self.device)

        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids,
                         7:13] = torch_rand_float(-0.5,
                                                  0.5,
                                                  (len(env_ids),
                                                   6),
                                                  device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(
            self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:,
                         7:9] = torch_rand_float(-max_vel,
                                                 max_vel,
                                                 (self.num_envs,
                                                  2),
                                                 device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))


    # def _update_terrain_curriculum(self, env_ids):
    #     """ Implements the game-inspired curriculum.

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     # Implement Terrain curriculum
    #     if not self.init_done:
    #         # don't change on initial reset
    #         return
    #     distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
    #     # robots that walked far enough progress to harder terains
    #     move_up = distance > self.terrain.env_length / 2
    #     # robots that walked less than half of their required distance go to
    #     # simpler terrains
    #     move_down = (distance < torch.norm(
    #         self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
    #     self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    #     # Robots that solve the last level are sent to a random one
    #     self.terrain_levels[env_ids] = torch.where(
    #         self.terrain_levels[env_ids] >= self.max_terrain_level, torch.randint_like(
    #             self.terrain_levels[env_ids], self.max_terrain_level), torch.clip(
    #             self.terrain_levels[env_ids], 0))  # (the minumum level is zero)
    #     self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
    #                                                      self.terrain_types[env_ids]]
    def _update_terrain_curriculum(self, env_ids):
        """Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s
            < (self.reward_scales["tracking_lin_vel"] / self.dt) * 0.5
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        mask = self.terrain_levels[env_ids] >= self.max_terrain_level
        self.success_ids = env_ids[mask]
        mask = self.terrain_levels[env_ids] < 0
        self.fail_ids = env_ids[mask]
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]
        # if self.cfg.commands.curriculum:
        #     self.command_ranges["lin_vel_x"][self.fail_ids, 0] = torch.clip(
        #         self.command_ranges["lin_vel_x"][self.fail_ids, 0] + 0.25,
        #         -self.cfg.commands.non_smooth_max_lin_vel_x,
        #         -0.5,
        #     )
        #     self.command_ranges["lin_vel_x"][self.fail_ids, 1] = torch.clip(
        #         self.command_ranges["lin_vel_x"][self.fail_ids, 1] - 0.25,
        #         0.5,
        #         self.cfg.commands.smooth_max_lin_vel_x,
        #     )
        #     self.command_ranges["lin_vel_y"][self.fail_ids, 0] = torch.clip(
        #         self.command_ranges["lin_vel_y"][self.fail_ids, 0] + 0.25,
        #         -self.cfg.commands.non_smooth_max_lin_vel_y,
        #         -0.5,
        #     )
        #     self.command_ranges["lin_vel_y"][self.fail_ids, 1] = torch.clip(
        #         self.command_ranges["lin_vel_y"][self.fail_ids, 1] - 0.25,
        #         0.5,
        #         self.cfg.commands.smooth_max_lin_vel_y,
        #     )


    # def update_command_curriculum(self, env_ids):
    #     """ Implements a curriculum of increasing commands

    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     # If the tracking reward is above 80% of the maximum, increase the
    #     # range of commands
    #     if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / \
    #             self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
    #         self.command_ranges["lin_vel_x"][0] = np.clip(
    #             self.command_ranges["lin_vel_x"][0] - 0.3,  self.cfg.commands.min_curriculum_x, 0.)
    #         self.command_ranges["lin_vel_x"][1] = np.clip(
    #             self.command_ranges["lin_vel_x"][1] + 0.3, 0., self.cfg.commands.max_curriculum_x)
    #     if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / \
    #             self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]:
    #         self.command_ranges["ang_vel_yaw"][0] = np.clip(
    #             self.command_ranges["ang_vel_yaw"][0] - 0.3,-self.cfg.commands.max_curriculum_yaw, 0.)
    #         self.command_ranges["ang_vel_yaw"][1] = np.clip(
    #             self.command_ranges["ang_vel_yaw"][1] + 0.3, 0., self.cfg.commands.max_curriculum_yaw)
    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if self.cfg.terrain.curriculum and len(self.success_ids) != 0:
            mask = (
                self.episode_sums["tracking_lin_vel"][self.success_ids]
                / self.max_episode_length
                > self.cfg.commands.curriculum_threshold
                * self.reward_scales["tracking_lin_vel"]
            )
            success_ids = self.success_ids[mask]
            slope_ids = torch.any(
                success_ids.unsqueeze(1) == self.smooth_slope_idx.unsqueeze(0), dim=1
            )
            slope_ids = success_ids[slope_ids]
            self.command_ranges["lin_vel_x"][success_ids, 0] -= 0.15
            self.command_ranges["lin_vel_x"][success_ids, 1] += 0.15
            self.command_ranges["lin_vel_x"][slope_ids, 0] -= 0.25
            self.command_ranges["lin_vel_x"][slope_ids, 1] += 0.25
            self.command_ranges["lin_vel_y"][success_ids, 0] -= 0.15
            self.command_ranges["lin_vel_y"][success_ids, 1] += 0.15
            self.command_ranges["lin_vel_y"][slope_ids, 0] -= 0.25
            self.command_ranges["lin_vel_y"][slope_ids, 1] += 0.25

            self.command_ranges["lin_vel_x"][self.smooth_slope_idx, :] = torch.clip(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, :],
                -1,
                self.cfg.commands.smooth_max_lin_vel_x,
            )
            self.command_ranges["lin_vel_y"][self.smooth_slope_idx, :] = torch.clip(
                self.command_ranges["lin_vel_y"][self.smooth_slope_idx, :],
                -1,
                self.cfg.commands.smooth_max_lin_vel_y,
            )
            self.command_ranges["lin_vel_x"][self.none_smooth_idx, :] = torch.clip(
                self.command_ranges["lin_vel_x"][self.none_smooth_idx, :],
                -1,
                self.cfg.commands.non_smooth_max_lin_vel_x,
            )
            self.command_ranges["lin_vel_y"][self.none_smooth_idx, :] = torch.clip(
                self.command_ranges["lin_vel_y"][self.none_smooth_idx, :],
                -self.cfg.commands.non_smooth_max_lin_vel_y,
                self.cfg.commands.non_smooth_max_lin_vel_y,
            )
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros(self.num_envs,45, dtype=torch.float, device=self.device, requires_grad=False)

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #     noise_vec[235:] = 0

        
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # Get the ground friction coefficient
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_root_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # get the rigid body states
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_root_state).view(self.num_envs, -1, 13) # shape: num_envs, num_bodies, 10 (pos, quat, lin_vel, ang_vel)
        print("body rigid body state shape: ", self.rigid_body_state.shape)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch(
            [1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.obs_history_length *self.cfg.env.num_observations, 
                                           dtype=torch.float, device=self.device)
        

        self.foot_positions = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_velocities = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs,self.num_dof,dtype=torch.float, device=self.device, requires_grad=False)
        self.trap_static_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_heights = torch.zeros_like(self.foot_positions)

        if self.cfg.domain_rand.randomize_action_delay:
            action_delay_idx = torch.round(
                torch_rand_float(
                    self.cfg.domain_rand.delay_ms_range[0] / 1000 / self.sim_params.dt,
                    self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt,
                    (self.num_envs, 1),
                    device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long()
            delay_max = np.int64(
                np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
            )
            self.action_fifo = torch.zeros(
                (self.num_envs, delay_max, self.cfg.env.num_actions),
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )

        self.joint_pos_target = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.Kp_factors = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.Kd_factors = torch.ones(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.power = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.p_gains = torch.zeros(
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.d_gains = torch.zeros(
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.last_last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(
            self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec)
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.v_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False)
        if self.cfg.domain_rand.randomize_imu_offset:
            min_angle, max_angle = self.cfg.domain_rand.randomize_imu_offset_range

            min_angle_rad = math.radians(min_angle)
            max_angle_rad = math.radians(max_angle)

            pitch = torch.rand(self.num_envs, device=self.device) * (max_angle_rad - min_angle_rad) + min_angle_rad
            roll = torch.rand(self.num_envs, device=self.device) * (max_angle_rad - min_angle_rad) + min_angle_rad

            pitch_quat = torch.stack(
                [torch.zeros_like(pitch), torch.sin(pitch / 2), torch.zeros_like(pitch), torch.cos(pitch / 2)], dim=-1)
            roll_quat = torch.stack(
                [torch.sin(roll / 2), torch.zeros_like(roll), torch.zeros_like(roll), torch.cos(roll / 2)], dim=-1)

            self.random_imu_offset = quat_mul(pitch_quat, roll_quat)
        else:
            self.random_imu_offset = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,1)
        self.command_ranges["lin_vel_x"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_x"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x
        )
        self.command_ranges["lin_vel_y"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["lin_vel_y"][:] = torch.tensor(
            self.cfg.commands.ranges.lin_vel_y
        )
        self.command_ranges["ang_vel_yaw"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["ang_vel_yaw"][:] = torch.tensor(
            self.cfg.commands.ranges.ang_vel_yaw
        )
        self.base_position = self.root_states[:, :3]
        self.last_base_position = self.base_position.clone()
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False) for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.k_factor = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.mass_params_tensor = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device, requires_grad=False)
        self.leg_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(
                robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            # print(body_props)
            # print(body_names)
            body_props, mass_params,leg_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)
            self.leg_params_tensor[i, :] = torch.from_numpy(leg_params).to(self.device).to(torch.float)
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
            self.restitution_tensor = self.restitutions.to(self.device).to(torch.float).squeeze(-1)
        self.feet_indices = torch.zeros(
            len(feet_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False)

        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False)

        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            
    def _randomize_dof_props(self, env_ids, cfg):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength

        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids,
                            :] = torch.rand(len(env_ids),
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False).unsqueeze(1) * (max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids,
                            :] = torch.rand(len(env_ids),
                                            dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False).unsqueeze(1) * (max_Kd_factor - min_Kd_factor) + min_Kd_factor
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids,
                               :] = torch.rand(len(env_ids),
                                               self.num_dof,
                                               dtype=torch.float,
                                               device=self.device,
                                               requires_grad=False) * (max_offset - min_offset) + min_offset
            
    # def _get_env_origins(self):
    #     """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
    #         Otherwise create a grid.
    #     """
    #     if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
    #         self.custom_origins = True
    #         self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    #         # put robots at the origins defined by the terrain
    #         max_init_level = self.cfg.terrain.max_init_terrain_level
    #         if not self.cfg.terrain.curriculum:
    #             max_init_level = self.cfg.terrain.num_rows - 1
    #         self.terrain_levels = torch.randint(
    #             0, max_init_level + 1, (self.num_envs,), device=self.device)
    #         self.terrain_types = torch.div(
    #             torch.arange(
    #                 self.num_envs,
    #                 device=self.device),
    #             (self.num_envs / self.cfg.terrain.num_cols),
    #             rounding_mode='floor').to(
    #             torch.long)

    #         self.max_terrain_level = self.cfg.terrain.num_rows
    #         self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
    #         self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
    #     else:
    #         self.custom_origins = False
    #         self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    #         # create a grid of robots
    #         num_cols = np.floor(np.sqrt(self.num_envs))
    #         num_rows = np.ceil(self.num_envs / num_cols)
    #         xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
    #         spacing = self.cfg.env.env_spacing
    #         self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
    #         self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
    #         self.env_origins[:, 2] = 0.
    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level + 1, (self.num_envs,), device=self.device
            )
            self.terrain_types = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
            )
            self.terrain_types[-self.num_envs :] = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            # num_cols = 20
            # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
            # terrain types: [0 1, 2 3, 4 5 6 7 8 9 10, 11 12 13 45 15, 16 17 18 19]
            # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
            self.smooth_slope_idx = (
                (self.terrain_types < 2).nonzero(as_tuple=False).flatten()
            )
            self.rough_slope_idx = (
                ((2 <= self.terrain_types) * (self.terrain_types < 4))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.stair_up_idx = (
                ((4 <= self.terrain_types) * (self.terrain_types < 11))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.stair_down_idx = (
                ((11 <= self.terrain_types) * (self.terrain_types < 16))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.discrete_idx = (
                ((16 <= self.terrain_types) * (self.terrain_types < 20))
                .nonzero(as_tuple=False)
                .flatten()
            )
            self.none_smooth_idx = torch.cat(
                (
                    self.rough_slope_idx,
                    self.stair_up_idx,
                    self.stair_down_idx,
                    self.discrete_idx,
                )
            )
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins)
                .to(self.device)
                .to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
            self.terrain_x_max = (
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_x_min = -self.cfg.terrain.border_size
            self.terrain_y_max = (
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_length
                + self.cfg.terrain.border_size
            )
            self.terrain_y_min = -self.cfg.terrain.border_size
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False
            )
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * \
            self.terrain.cfg.vertical_scale
    def _get_foot_heights(self):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = self.foot_positions[:, :, :2] + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        # heights = torch.zeros_like(self.height_samples[px, py])
        # for i in range(2):
        #     for j in range(2):
        #         heights += self.height_samples[px + i - 1, py + j - 1]
        # heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale / 9

        return heights
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target) * 20
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_power(self):
        # Penalize power
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square(
                (self.last_dof_vel - self.dof_vel) / self.dt),
            dim=1)


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - 2.* self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum(
    #         (torch.abs(
    #             self.dof_vel) -
    #             self.dof_vel_limits *
    #             self.cfg.rewards.soft_dof_vel_limit).clip(
    #             min=0.,
    #             max=1.),
    #         dim=1)
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        dof_vel_limits = torch.clip(10*self.v_level.unsqueeze(-1).repeat(1,self.num_dof), min=10, max=20)
        error = torch.sum((torch.abs(self.dof_vel) - dof_vel_limits).clip(min=0., max=15.), dim=1)
        # print("dof_vel",self.dof_vel[:,[1,2]])
        rew = 1 - torch.exp(-1 * error)
        return rew
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        L_torque_sum = torch.sum(torch.abs(self.torques[:,[0,1,2]]), dim=1)
        R_torque_sum = torch.sum(torch.abs(self.torques[:,[6,7,8]]), dim=1)

        # print("torque",L_torque_sum,R_torque_sum)

        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0., max=1.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        print("vel",self.base_lin_vel[:, :1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)


    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is
    #     # unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # reward only on first contact with the ground
        des_feet_air_time = 0.5/self.v_level.unsqueeze(-1).repeat(1,self.feet_indices.shape[0])
        # des_feet_air_time = 0.2
        rew_airTime = torch.sum(torch.clamp((self.feet_air_time - des_feet_air_time), max=0.) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        cmd_mask = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > self.cfg.commands.min_vel, 
        torch.abs(self.commands[:, 2]) > self.cfg.commands.min_vel)
        rew_airTime[~cmd_mask] = -torch.sum(self.feet_air_time[~cmd_mask], dim=1) # reward stand still for zero command
        return rew_airTime
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:,self.feet_indices,:2],
                                    dim=2) > 3 * torch.abs(self.contact_forces[:,self.feet_indices,2]),dim=1)


    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos),
                         dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.05
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        return reward
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 2.)*\
                  (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        # print("stumble:",stumble)
        return torch.sum(stumble, dim=1)
    



    def _reward_trap_static(self):
        lin_trap_static = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < self.cfg.commands.min_vel * 0.5)*(
            torch.norm(self.commands[:, :2], dim=-1) > self.cfg.commands.min_vel)
        ang_trap_static = (torch.abs(self.base_ang_vel[:, 2]) < self.cfg.commands.min_vel * 0.5)*(
            torch.abs(self.commands[:, 2]) > self.cfg.commands.min_vel)
        trap_mask = torch.logical_or(lin_trap_static, ang_trap_static)
        self.trap_static_time[trap_mask] += self.dt
        self.trap_static_time[~trap_mask] = 0.
        # print("trap_static_time:",self.trap_static_time)
        self.trap_static_time.clip(max=5.)
        # foot_height = torch.mean(self.foot_positions[:, :, 2].unsqueeze(2) - self.foot_measured_heights, dim=-1)

        # foot_height = torch.clamp(foot_height-0.022,0,1)
        # foot_height = torch.sum(torch.clamp(torch.abs(0.01-foot_height),0,0.1), dim=-1)*100
        reward = self.trap_static_time
        return reward
    
    def _reward_hip_limit(self):
        reward = torch.sum(torch.square(self.dof_pos[:,[0,3,6,9]] - self.default_dof_pos[:,[0,3,6,9]]),dim=1)
        # print("hip limit",reward)
        return reward
    def _reward_trot_gait(self):

        return torch.sum(torch.square(self.dof_pos[:,[1,2]] - self.dof_pos[:,[10,11]])+
                         torch.square(self.dof_pos[:,[4,5]] - self.dof_pos[:,[7,8]]), dim=-1)
    def _reward_power_distribution(self):
        thigh_power = torch.abs(torch.var(self.torques[:,[1,4,7,10]]*self.dof_vel[:,[1,4,7,10]]))
        calf_power = torch.abs(torch.var(self.torques[:,[2,5,8,11]]*self.dof_vel[:,[2,5,8,11]]))

        return torch.sum(thigh_power + calf_power,dim=-1)