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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2BackflipCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        # additional: base_lin_vel, external_forces, scan_dots
        # has_jumped added dynamically
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 + 1  # 238 total
        # priviledge_obs_buf returned by step() for asymmetric training
        # +1 for has_jumped
        num_privileged_obs = (num_one_step_privileged_obs) * 1
        num_actions = 12
        # not used with heightfields/trimeshes
        env_spacing = 3.
        # send time out information to the algorithm
        send_timeouts = True
        # episode length in seconds
        episode_length_s = 20

        reset_height = 0.1
        reset_landing_error = 0.2 # [in m]
        reset_orientation_error=0.8 # [in rad]
        test = False

    class terrain(LeggedRobotCfg.terrain):
        # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'plane'

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

        start_joint_angles = { # = target angles [rad] when action = 0.0

            'FL_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': -0. ,  # [rad]
            'RR_hip_joint': -0.,   # [rad]

            'FL_thigh_joint': 0.9,     # [rad]
            'RL_thigh_joint': 1.,#1.,   # [rad]
            'FR_thigh_joint': 0.9,     # [rad]
            'RR_thigh_joint': 1.,#1.,   # [rad]

            'FL_calf_joint': -2.25,   # [rad]
            'RL_calf_joint': -2.25,    # [rad]
            'FR_calf_joint': -2.25,  # [rad]
            'RR_calf_joint': -2.25,    # [rad]
        }
        

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1

    class commands( LeggedRobotCfg.commands ):
            curriculum = False
            max_curriculum = 0.8
            num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 5. # time before command are changed[s]
            heading_command = False # if true: compute ang vel command from heading error
            class ranges: 
                pos_dx_lim = [-0.0,0.8]
                pos_dy_lim = [-0.1,0.1]
                pos_dz_lim = [-0.0,0.8]
                # These are the steps for the jump distance changes every curriculum update.
                pos_variation_increment = [0.01,0.01,0.01]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Yun1/urdf/yun1.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            before_setting=5.0
            # setting=5.0
            line_z=12.
            angle_y=2.
            base_height_flight=10.0
            base_height_stance=20.0
            orientation=25.0
            dof_pose_air=-0.2
            dof_pos_stance=20.0
            ang_vel_xy=-0.05
            torques=-0.0001
            dof_pos_limits=-5.
            dof_vel_limits=-5.
            dof_vel=-0.001
            torque_limits=-1.
            termination=0.0
            collision=-10.
            action_rate=-0.005
            feet_contact_forces=-0.1
            symmetric_joints=-0.5
            dof_hip_pos=-10.   

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        # soft_dof_vel_limit = 0.9
        # soft_torque_limit = 0.9
        cycle_time=1.0
        target_height=0.6
        max_contact_force = 150. # forces above this value are penalized

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_towards_goal=True
        
        # # 基座位置随机化
        # randomize_base_init_pos = True
        # base_init_pos_range_x = [-0.2, 0.2]  # x方向随机偏移 ±20cm
        # base_init_pos_range_y = [-0.2, 0.2]  # y方向随机偏移 ±20cm  
        # base_init_pos_range_z = [0.35, 0.5]  # z高度在35-50cm范围内随机
        
        # # 基座姿态随机化（欧拉角，弧度）
        # randomize_base_init_orientation = True
        # base_init_roll_range = [-0.5, 0.5]   # 横滚角 ±11.5度
        # base_init_pitch_range = [-0.5, 0.5]  # 俯仰角 ±11.5度
        # base_init_yaw_range = [-0.5, 0.5]    # 偏航角 ±28.6度
        
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

class Go2BackflipCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        runner_class_name = 'HIMOnPolicyRunner'  # 现在在正确位置
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        max_iterations = 10000 # number of policy updates
        run_name = ''
        experiment_name = 'backflip_go2'
        num_steps_per_env = 24 # per iteration
