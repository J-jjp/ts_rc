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
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE+++
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:

        num_envs = 4096*2
        num_observations = 45 # proprioceptive
        # if not None a priviledge_obs_buf will be returned by step() (critic
        # obs for assymetric training). None is returned otherwise
        # proprio + lin vel + terrrain + doman_random + contact force  + joint torques + joint accelaration 
        num_privileged_obs = 45 + 3 + 187 + 38 + 12 + 12+12
        num_privileged_latent = 32
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        num_actors = 2  # 2 actors
        obs_history_length = 15  # number of observations to stack
        fail_to_terminal_time_s = 0.5


    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                             0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -
                             0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stone, gap, pit]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.2, 0.2, 0.1, 0.1, 0.4, 0.0, 0.0, 0.0]
        terrain_proportions = [0.1, 0.2,0.3, 0.2, 0.2]
        height = [0.02, 0.03]
        downsampled_scale = 0.05
        # trimesh only:
        # slopes above this threshold will be corrected to vertical surfaces
        slope_treshold = 0.75

    class commands:
        # curriculum = True
        # min_curriculum_x = -1.

        # max_curriculum_x = 1
        # max_curriculum_yaw = 2.
        curriculum = True
        smooth_max_lin_vel_x = 4
        smooth_max_lin_vel_y = 1
        non_smooth_max_lin_vel_x = 3
        non_smooth_max_lin_vel_y = 1
        max_ang_vel_yaw = 2
        curriculum_threshold = 0.75
        zero_command_prob=0
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode
        # ang_vel_yaw is recomputed from heading error)
        min_vel = 0.15
        num_commands = 4
        resampling_time = 5.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.75, 0.75]  # min max [m/s]
            lin_vel_y = [-0.75, 0.75]  # min max [m/s]
            ang_vel_yaw = [-2., 2.]    # min max [rad/s]
            heading = [-3.14, 3.14]

        # class ranges:
        #     lin_vel_x = [-0.001, 0.001]   # min max [m/s]
        #     lin_vel_y = [-0.001, 0.001]   # min max [m/s]
        #     ang_vel_yaw = [-0.001, 0.001]    # min max [rad/s]
        #     heading = [-0.002, 0.002]
        
    class init_state:
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y        curriculum = True


    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction=0.5  # scale down hip flexion range
        max_power = 1000.0  # [W]

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        # name of the feet bodies, used to index body state and contact force
        # tensors
        foot_name = "None"
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be
        # kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True
        fix_base_link = False  # fixe the base of the robot
        # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3
        # effort)
        default_dof_drive_mode = 3
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        # replace collision cylinders with capsules, leads to faster/more
        # stable simulation
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_base_inertia = False
        added_inertia_range_xx = [-0.02,0.02]
        added_inertia_range_xy = [-0.0002,0.0002]
        added_inertia_range_xz = [-0.0002,0.0002]
        added_inertia_range_yy = [-0.02,0.02]
        added_inertia_range_zz = [-0.02,0.02]

        randomize_leg_mass = True
        added_leg_mass_range = [-0.2,0.2]
        factor_leg_mass_range = [0.85,1.15]

        
        randomize_leg_com = True
        added_leg_com_range = [-0.015, 0.015]
        
        randomize_friction = True
        friction_range = [0.05, 2.25]
        restitution_range = [0.0, 0.5]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 1.
        
        
        randomize_base_com = True
        added_com_range = [-0.05, 0.05]
        
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]

        randomize_action_delay = True
        delay_ms_range = [0, 20] # ms
        
        randomize_motor_offset = True
        motor_offset_range = [-0.05, 0.05]

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
    class rewards:
        class scales:
            termination = -0.
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.065
            orientation = -0.05
            torques = -1e-4
            power = -1e-5
            dof_vel = -1e-4
            stumble = -0.02
            feet_regulation = -0.07
            dof_acc = -2.5e-7
            base_height = -2.
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.2
            trap_static = -2.
            hip_limit = -0.75

            action_rate = -0.01
            action_smoothness = -0.005
            stand_still = -0.05

            dof_pos_limits = -2.0
            dof_vel_limits = -1
            torque_limits = -2

            power_distribution = -1e-5
            trot_gait = -0.02

        # if true negative total rewards are clipped at zero (avoids early
        # termination problems)
        only_positive_rewards = False
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.8  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8

        max_contact_force = 100.  # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0


        clip_observations = 100.
        clip_actions =10

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            adaptive_noise = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [100, 50, 30]  # [m]
        lookat = [50., 50, 3.]  # 

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 14000  # number of policy updates

        # logging
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
        student_reinforcing = False
