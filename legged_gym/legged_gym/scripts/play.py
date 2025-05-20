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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import time
import rospy
from sensor_msgs.msg import Joy
joy_cmd = [0.0, 0.0, 0.0]
is_teacher =True
def joy_callback(joy_msg):
    global joy_cmd
    global stop
    global begin
    joy_cmd[0] =  joy_msg.axes[1]
    joy_cmd[1] =  joy_msg.axes[0]
    joy_cmd[2] =  joy_msg.axes[3]  # 横向操作


def play(args):
    # get the environment and training configuration
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs,1)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    #env_cfg.viewer.pos = [20, 0, 3]  # [m]
    #env_cfg.viewer.lookat = [20., 10, 2.]  # [m]

    # env_cfg.domain_rand.randomize_motor = True
    # env_cfg.domain_rand.randomize_base_com = True
    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)
    # prepare environment
    path_1 = "/home/ubuntu/isaac/t_s/quadruped_rl_blind_walk/legged_gym/logs/go2_load_teacher_student_phase_model_a/May20_13-43-03_reinforce/model_5500.pt"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()  # get initial observations
    obs_history = env.get_observations_history()
    privileged_obs = env.get_privileged_observations()
    # estimated_obs = env.get_estimated_observations()
    # load policy 

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,path= path_1)
    policy = ppo_runner.get_inference_policy(
        device=env.device)  # get the policy

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        # path = os.path.join(
        #     LEGGED_GYM_ROOT_DIR,
        #     'logs',
        #     train_cfg.runner.experiment_name,
        #     'exported',
        #     'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path_1)
        print('Exported policy as jit script to: ', path_1)

    logger = Logger(env.dt)  # create a logger to log states and rewards
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 10  # number of steps before plotting states
    # number of steps before print average episode rewards
    env.max_episode_length=100*100
    stop_rew_log = env.max_episode_length
    camera_position = np.array(
        env_cfg.viewer.pos, dtype=np.float64)  # camera position
    camera_vel = np.array([1., 1., 0.])  # camera velocity
    camera_direction = np.array(
        env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10 * int(env.max_episode_length)):
        # latent = ppo_runner.alg.actor_critic.privileged_encoder(
        #     privileged_obs.detach())
        # print("latent:", latent)
        # obs = obs.detach()
        # latent = latent.detach()
        if joy_cmd[0]>0:
            env.commands[:, 0] = joy_cmd[0]*3.6
        else:
            env.commands[:, 0] = joy_cmd[0]
        env.commands[:, 1] = joy_cmd[1]*1
        env.commands[:, 2] = joy_cmd[2]*2
        actions = policy(obs.detach(), obs_history.detach(), privileged_obs.detach())  # get actions from the policy
        obs, privileged_obs , rews, dones, infos, obs_history = env.step(actions.detach())  # step the environment
        # actions = policy(obs.detach(), obs_history)  # get actions from the policy
        # obs, _, rews, dones, infos, obs_history = env.step(
        #     actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    'logs',
                    train_cfg.runner.experiment_name,
                    'exported',
                    'frames',
                    f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position,
                            camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.task = "go2"
    play(args)
